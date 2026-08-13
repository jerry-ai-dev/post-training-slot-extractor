# Multi-turn Comparison Design

## Goal

Keep prior turns visible and pass an independent conversation history to each model on later comparisons.

## Design

The browser owns the demo session. Before each request it snapshots the current input, appends a visible user-turn marker to both columns, and sends the existing `histories.left` and `histories.right`. When a side receives a non-empty reply, the browser appends that turn's user and assistant messages to only that side's history. This matches the natural-language history shape already accepted by the orchestrator and training data.

Trace events and replies append to the existing column instead of replacing it. A turn divider groups each new user message with the events that follow. A new explicit `清空对话` control resets both histories, exported events, column contents, and side statuses. Model selectors remain unchanged.

If a model errors or returns a null reply, that unsuccessful exchange is displayed but not committed to future model context. The two sides therefore remain independent even if one succeeds and the other fails.

## Tests

Static integration contracts verify that the run handler no longer clears conversation columns, that replies update side-specific history with user and assistant messages, that user-turn markers are appended, and that the reset control clears histories and display state.
