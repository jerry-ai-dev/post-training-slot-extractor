# Progressive Comparison UI Design

## Goal

Make sequential comparison visibly progress one model at a time. Each column must show its own loading and inference state, render its result as soon as that side finishes, and report model inference time without including server startup or shutdown. Expanding intermediate events must never break the two-column layout.

## Event flow

`POST /api/compare` remains an NDJSON streaming endpoint. While holding the existing comparison lock, it yields lifecycle events as work happens instead of accumulating both sides before constructing the response.

For each side in sequential order:

1. Emit `side_status` with `status=loading` before starting llama-server.
2. Emit `side_status` with `status=inferencing` immediately before invoking the conversation orchestrator.
3. Run the complete tool loop and measure only calls into the generation backend. If a tool loop invokes the model more than once, sum those generation durations.
4. Stream the side's existing trace events.
5. Emit `side_status` with `status=complete` and `inference_duration_ms`.
6. Continue with the other side.

Errors remain visible as trace events. A side that fails emits an error status and does not falsely report successful completion.

## Backend structure

The comparison implementation becomes a generator consumed by `StreamingResponse`. The generator owns the comparison lock for the full model lifecycle so multiple browser tabs cannot collide on the shared llama-server port.

A small timing backend decorator wraps the selected inference backend and accumulates wall-clock time only around `generate()`. Model manifest verification, process startup, readiness checks, and process shutdown are outside the measured interval.

## Frontend behavior

Each model column receives a persistent status panel. On a new run both columns reset to waiting. In sequential mode:

- Left changes from loading to inferencing, then shows completion and inference time.
- Right remains waiting until its first lifecycle event, then follows the same states.
- Trace and reply events render immediately as NDJSON lines arrive.

The global button remains disabled for the complete comparison to prevent duplicate requests. The global status summarizes the entire run; it does not replace per-column status.

## Layout resilience

Grid children and model cards use `min-width: 0`. Event details and `pre` blocks are constrained to their column, use wrapping for long JSON tokens, and retain horizontal scrolling as a fallback. Opening and closing any details element therefore cannot change the grid from two columns on desktop.

## Testing

Integration tests verify:

- Lifecycle events arrive in left-then-right order.
- Left completion is yielded before right inference starts.
- Reported inference time covers backend generation but excludes simulated model setup.
- Frontend contains per-side state handling and inference-time display.
- CSS includes the width, wrapping, and overflow constraints required to preserve the grid.
- Comparison serialization remains enforced.

Real verification runs the two SFT models through the endpoint and checks lifecycle order, two successful completions, and nonzero inference durations. The running app is restarted on port 8001 after implementation.
