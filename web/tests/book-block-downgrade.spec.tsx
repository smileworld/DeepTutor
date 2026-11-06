import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import BlockRenderer from "@/app/(workspace)/learning/books/components/blocks/BlockRenderer";
import { initI18n } from "@/i18n/init";
import type { Block } from "@/lib/book-types";

initI18n("en");

vi.mock("@/components/common/MarkdownRenderer", () => ({
  default: ({ content }: { content: string }) => <div>{content}</div>,
}));

function makeBlock(metadata: Record<string, unknown> = {}): Block {
  return {
    id: "blk_downgrade",
    type: "text",
    status: "ready",
    title: "",
    params: {},
    payload: { body: "Generated prose" },
    source_anchors: [],
    metadata,
    error: "",
    created_at: 0,
    updated_at: 0,
  } as Block;
}

describe("automatic block downgrade notice", () => {
  it("renders a durable notice with the original type and reason", () => {
    const { container } = render(
      <BlockRenderer
        block={makeBlock({
          downgrade: {
            original_type: "animation",
            downgraded_to: "text",
            reason: "math-animator extras are not installed",
          },
        })}
      />,
    );

    expect(
      screen.getByText(/Downgraded from animation to text/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/math-animator extras are not installed/i),
    ).toBeInTheDocument();
    expect(
      container.querySelector('[data-block-downgraded="animation"]'),
    ).not.toBeNull();
  });

  it("stays out of the way for a block that was never downgraded", () => {
    const { container } = render(<BlockRenderer block={makeBlock()} />);

    expect(screen.queryByText(/Downgraded from/i)).toBeNull();
    expect(container.querySelector("[data-block-downgraded]")).toBeNull();
  });

  it("uses the current block type when the manifest has no target", () => {
    render(
      <BlockRenderer
        block={makeBlock({ downgrade: { original_type: "quiz" } })}
      />,
    );

    expect(screen.getByText(/Downgraded from quiz to text/i)).toBeInTheDocument();
  });
});
