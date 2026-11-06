"""L3 — block capability probing and traceable automatic downgrade.

These are the four manual ``change-block-type`` repairs from the failing book
turned into system behaviour: animation/interactive -> section, quiz ->
flash_cards. No LLM is called anywhere in this module.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from deeptutor.book import compiler as compiler_module
from deeptutor.book.blocks import animation as animation_module
from deeptutor.book.blocks.base import (
    BlockContext,
    BlockGenerator,
    BlockGeneratorRegistry,
    GenerationFailure,
    ProbeResult,
    get_block_registry,
)
from deeptutor.book.blocks.capabilities import (
    DEFAULT_DOWNGRADE_MAP,
    capabilities_snapshot,
    clear_probe_cache,
    downgrade_map_for_book,
    probe_registry,
    record_capabilities,
)
from deeptutor.book.compiler import BookCompiler, CompilerOptions, systemic_failure_reason
from deeptutor.book.models import (
    Block,
    BlockStatus,
    BlockType,
    Book,
    Chapter,
    ContentType,
    Page,
    PageStatus,
)


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    clear_probe_cache()
    yield
    clear_probe_cache()


class _Storage:
    def __init__(self, book: Book | None = None) -> None:
        self.book = book or Book(id="bk_l3")
        self.saved_pages: list[Page] = []
        self.logs: list[str] = []

    def load_book(self, book_id: str) -> Book | None:
        return self.book if book_id == self.book.id else None

    def save_book(self, book: Book) -> None:
        self.book = book.model_copy(deep=True)

    def save_page(self, page: Page) -> None:
        self.saved_pages.append(page)

    def append_log(self, book_id: str, message: str, **_kwargs) -> None:
        self.logs.append(message)


class _Stream:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    @asynccontextmanager
    async def stage(self, *_args, **_kwargs):
        yield

    async def book_event(self, kind: str, payload: dict, **_kwargs) -> None:
        self.events.append((kind, payload))


class _FakeGenerator(BlockGenerator):
    def __init__(
        self,
        block_type: BlockType,
        *,
        fail: str | None = None,
        probe_result: ProbeResult | None = None,
    ) -> None:
        self.block_type = block_type
        self.fail = fail
        self.probe_result = probe_result
        self.calls = 0

    async def probe(self, ctx: BlockContext | None = None) -> ProbeResult:
        return self.probe_result or ProbeResult()

    async def _generate(self, ctx: BlockContext):
        self.calls += 1
        if self.fail:
            raise GenerationFailure(self.fail)
        return ({"text": "generated"}, [], {})


class _RaisingProbe(_FakeGenerator):
    async def probe(self, ctx: BlockContext | None = None) -> ProbeResult:
        raise RuntimeError("probe exploded")


def _registry(*generators: BlockGenerator) -> BlockGeneratorRegistry:
    registry = BlockGeneratorRegistry()
    for generator in generators:
        registry.register(generator)
    return registry


def _compiler(
    storage: _Storage,
    registry: BlockGeneratorRegistry,
    *,
    auto_downgrade: bool = True,
    capability_probe: bool = True,
) -> BookCompiler:
    compiler = BookCompiler(
        storage=storage,
        options=CompilerOptions(
            phase=2,
            block_retry_attempts=0,
            auto_downgrade=auto_downgrade,
            capability_probe=capability_probe,
        ),
    )
    compiler.registry = registry
    from deeptutor.book.agents.page_planner import SectionArchitect

    compiler.architect = SectionArchitect(phase=2, llm_enabled=False)
    return compiler


def _chapter() -> Chapter:
    return Chapter(
        id="ch_l3",
        title="Capability chapter",
        content_type=ContentType.THEORY,
    )


def _page(*blocks: Block) -> Page:
    return Page(
        id="pg_l3",
        book_id="bk_l3",
        chapter_id="ch_l3",
        title="A page",
        blocks=list(blocks),
    )


async def _compile(
    compiler: BookCompiler, page: Page, stream: _Stream | None = None
) -> Page:
    return await compiler.compile_page(
        book_id="bk_l3",
        chapter=_chapter(),
        page=page,
        stream=stream or _Stream(),
    )


# ── Probing ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_default_probe_is_available() -> None:
    assert (await _FakeGenerator(BlockType.SECTION).probe()).available is True


@pytest.mark.asyncio
async def test_registry_probe_all_isolates_a_raising_probe() -> None:
    registry = _registry(
        _FakeGenerator(BlockType.SECTION),
        _RaisingProbe(BlockType.FIGURE),
        _FakeGenerator(BlockType.QUIZ, probe_result=ProbeResult(False, "no extras")),
    )
    results = await registry.probe_all()
    assert results[BlockType.SECTION].available is True
    assert results[BlockType.FIGURE].available is False
    assert "probe exploded" in results[BlockType.FIGURE].reason
    assert results[BlockType.QUIZ].reason == "no extras"
    assert await registry.available_types() == {BlockType.SECTION}


@pytest.mark.asyncio
async def test_animation_probe_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = animation_module.AnimationGenerator()
    monkeypatch.setattr(animation_module, "_missing_dependency", lambda: "manim missing")
    result = await generator.probe()
    assert result.available is False
    assert result.reason == "manim missing"

    monkeypatch.setattr(animation_module, "_missing_dependency", lambda: None)
    assert (await generator.probe()).available is True


def test_missing_dependency_detects_disabled_module_and_missing_manim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deeptutor.modules import catalog

    monkeypatch.setattr(catalog, "require_module", lambda _name: None)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    assert "math-animator" in (animation_module._missing_dependency() or "")

    monkeypatch.setattr("importlib.util.find_spec", lambda _name: object())
    assert animation_module._missing_dependency() is None


@pytest.mark.asyncio
async def test_animation_generate_still_guards_when_cache_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(animation_module, "_missing_dependency", lambda: "install extras")
    ctx = BlockContext(
        book_id="bk_l3",
        chapter=_chapter(),
        page=_page(),
        block=Block(type=BlockType.ANIMATION),
    )
    with pytest.raises(GenerationFailure, match="install extras"):
        await animation_module.AnimationGenerator()._generate(ctx)


def test_real_default_registry_exposes_probe() -> None:
    registry = get_block_registry()
    assert BlockType.ANIMATION in registry.types()
    assert callable(getattr(registry.get(BlockType.ANIMATION), "probe", None))


# ── Downgrade policy ────────────────────────────────────────────────────


def test_downgrade_map_defaults_and_overrides() -> None:
    assert downgrade_map_for_book(Book()) == DEFAULT_DOWNGRADE_MAP
    assert downgrade_map_for_book(Book(metadata={"auto_downgrade": False})) == {}

    custom = downgrade_map_for_book(
        Book(
            metadata={
                "block_downgrades": {
                    "quiz": "callout",
                    "animation": None,
                    "nonsense": "section",
                    "interactive": "not-a-type",
                }
            }
        )
    )
    assert custom[BlockType.QUIZ] == BlockType.CALLOUT
    assert BlockType.ANIMATION not in custom
    # An invalid target must not silently remove an existing default.
    assert custom[BlockType.INTERACTIVE] == BlockType.SECTION


def test_capabilities_snapshot_is_stable() -> None:
    registry = _registry(
        _FakeGenerator(BlockType.SECTION),
        _FakeGenerator(BlockType.QUIZ, probe_result=ProbeResult(False, "slow")),
    )

    async def run() -> None:
        results = await probe_registry(registry)
        first = capabilities_snapshot(results, registry)
        second = capabilities_snapshot(results, registry)
        assert first == second
        assert first["unavailable"] == {"quiz": "slow"}
        assert "section" in first["available"]

    import asyncio

    asyncio.run(run())


def test_record_capabilities_is_best_effort() -> None:
    book = Book(id="bk_l3")
    snapshot = {"fingerprint": "abc", "probed_at": 1.0}
    record_capabilities(book, object(), snapshot)  # no save_book → must not raise
    assert book.metadata["capabilities"] == snapshot
    record_capabilities(book, object(), snapshot)  # idempotent
    assert book.metadata["capabilities"] == snapshot


# ── Planning filter ─────────────────────────────────────────────────────


def _plannable_registry() -> BlockGeneratorRegistry:
    return _registry(
        _FakeGenerator(BlockType.SECTION),
        _FakeGenerator(BlockType.TEXT),
        _FakeGenerator(BlockType.CALLOUT),
        _FakeGenerator(BlockType.FIGURE),
        _FakeGenerator(
            BlockType.QUIZ, probe_result=ProbeResult(False, "quiz times out")
        ),
        _FakeGenerator(BlockType.FLASH_CARDS),
        _FakeGenerator(BlockType.INTERACTIVE),
        _FakeGenerator(BlockType.CODE),
        _FakeGenerator(BlockType.TIMELINE),
    )


@pytest.mark.asyncio
async def test_planning_omits_unavailable_block_types() -> None:
    storage = _Storage()
    compiler = _compiler(storage, _plannable_registry())
    page = _page()

    stream = _Stream()
    await compiler._plan_if_needed(
        "bk_l3", _chapter(), page, stream, language="en"
    )

    assert BlockType.QUIZ not in {block.type for block in page.blocks}
    assert BlockType.SECTION in {block.type for block in page.blocks}
    assert "quiz" in storage.book.metadata["capabilities"]["unavailable"]


@pytest.mark.asyncio
async def test_planning_unfiltered_when_probe_disabled() -> None:
    storage = _Storage()
    compiler = _compiler(storage, _plannable_registry(), capability_probe=False)
    page = _page()

    await compiler._plan_if_needed("bk_l3", _chapter(), page, _Stream(), language="en")

    assert BlockType.QUIZ in {block.type for block in page.blocks}
    assert "capabilities" not in storage.book.metadata


# ── Runtime downgrade ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_quiz_downgrades_to_flash_cards_and_clears_breaker() -> None:
    flash = _FakeGenerator(BlockType.FLASH_CARDS)
    registry = _registry(
        _FakeGenerator(BlockType.QUIZ, fail="quiz generation failed: timed out"),
        flash,
    )
    storage = _Storage()
    compiler = _compiler(storage, registry)
    stream = _Stream()
    page = _page(Block(type=BlockType.QUIZ))

    result = await _compile(compiler, page, stream)
    block = result.blocks[0]

    assert block.type == BlockType.FLASH_CARDS
    assert block.status == BlockStatus.READY
    assert block.payload == {"text": "generated"}
    assert block.metadata["downgrade"]["original_type"] == "quiz"
    assert block.metadata["downgrade"]["downgraded_to"] == "flash_cards"
    assert "timed out" in block.metadata["downgrade"]["reason"]
    assert "failure" not in block.metadata
    assert result.status == PageStatus.READY
    assert systemic_failure_reason(result) == "", (
        "a recovered block must not trip the consecutive-failure breaker"
    )
    assert any(kind == "block_downgraded" for kind, _ in stream.events)
    assert flash.calls == 1


@pytest.mark.asyncio
async def test_already_downgraded_block_is_not_downgraded_again() -> None:
    flash = _FakeGenerator(BlockType.FLASH_CARDS)
    registry = _registry(
        _FakeGenerator(BlockType.QUIZ, fail="quiz failed again"), flash
    )
    compiler = _compiler(_Storage(), registry)
    block = Block(
        type=BlockType.QUIZ,
        metadata={"downgrade": {"original_type": "animation"}},
    )
    page = _page(block)

    result = await _compile(compiler, page)

    assert block.type == BlockType.QUIZ
    assert block.status == BlockStatus.ERROR
    assert flash.calls == 0, "no downgrade chains / cycles"


@pytest.mark.asyncio
async def test_kill_switch_keeps_the_original_error() -> None:
    registry = _registry(
        _FakeGenerator(BlockType.QUIZ, fail="quiz generation failed: timed out"),
        _FakeGenerator(BlockType.FLASH_CARDS),
    )
    compiler = _compiler(_Storage(), registry, auto_downgrade=False)
    page = _page(Block(type=BlockType.QUIZ))

    result = await _compile(compiler, page)

    assert result.blocks[0].type == BlockType.QUIZ
    assert result.blocks[0].status == BlockStatus.ERROR
    assert "downgrade" not in result.blocks[0].metadata
    assert systemic_failure_reason(result).startswith("timeout:")


@pytest.mark.asyncio
async def test_per_book_downgrade_override_is_honoured() -> None:
    registry = _registry(
        _FakeGenerator(BlockType.QUIZ, fail="quiz generation failed: timed out"),
        _FakeGenerator(BlockType.CALLOUT),
        _FakeGenerator(BlockType.FLASH_CARDS),
    )
    storage = _Storage(
        Book(id="bk_l3", metadata={"block_downgrades": {"quiz": "callout"}})
    )
    compiler = _compiler(storage, registry)
    page = _page(Block(type=BlockType.QUIZ))

    result = await _compile(compiler, page)
    assert result.blocks[0].type == BlockType.CALLOUT
    assert result.blocks[0].status == BlockStatus.READY


def test_compiler_options_expose_the_kill_switches() -> None:
    options = CompilerOptions()
    assert options.auto_downgrade is True
    assert options.capability_probe is True
    assert compiler_module.SYSTEMIC_FAILURE_KINDS == frozenset(
        {"rate_limit", "provider_error", "timeout"}
    )
