# PCD Server product README design

**Date:** 2026-09-18  
**Status:** Approved

## Goal

Turn the repository README into a product-first landing page and developer guide that explains the value of PCD Server, shows the embedded web UI, and gets a developer from clone to a successful REST request quickly.

## Audience and positioning

The primary audience is developers evaluating or integrating a local structured-classification service. The opening should describe the outcome in concrete terms: PCD Server turns text into guaranteed bounded JSON using a local GGUF model. Parallel Constrained Decoding terminology and implementation details remain available later for readers who want to understand how it works.

The README documents the existing native CMake workflow only. It must not imply that Docker images, package-manager releases, authentication, arbitrary JSON Schema, or multi-model residency exist.

## Content architecture

Use progressive disclosure in this order:

1. **Hero** — product name, concise value proposition, restrained badges, key capability summary, and the strongest Playground screenshot.
2. **Why PCD Server** — always-valid bounded output, local GGUF inference, per-choice probabilities, and schema-prefix caching.
3. **Quick start** — prerequisites and copy-paste commands to download the default model, configure, build, test, and run.
4. **First decode** — a complete request and compact representative response, with immediate links to the Playground and API documentation.
5. **Web UI tour** — screenshots of the Playground, scenario-driven extraction, Tetris visualization, and Swagger/OpenAPI documentation.
6. **REST API** — endpoint summary, decode request and response semantics, model discovery and switching, errors, and limits.
7. **Operations and development** — server flags, model precedence, tests, benchmark usage, concurrency, and supported platforms.
8. **How it works** — retain the detailed decode pipeline and checkpoint rationale below the getting-started path.
9. **Scope** — clearly list current supported and unsupported capabilities.

## Visual assets

Commit selected browser captures under `docs/images/` with descriptive file names and meaningful Markdown alt text. Use a full-width Playground result as the main product image. Use additional images for the scenario selector, Tetris visualization, and API docs without overwhelming the page.

The source captures currently live in the ignored `.playwright-mcp/` directory. Select clean captures that show successful output and avoid captures containing transient error states. README links must use repository-relative paths so they render on Git hosting.

## Copy and presentation

- Use GitHub-native Markdown with a clear hierarchy, small tables, and concise callouts.
- Prefer short, outcome-oriented copy before technical terminology.
- Keep badges useful and restrained; avoid badges for unsupported CI or release channels.
- Make the native quick start runnable as one sequential command block.
- Label latency and checkpoint-size figures as example Apple Silicon measurements rather than guarantees.
- Preserve the existing details about bounded booleans/string enums, schema cache status, response probabilities, context limits, transactional model switching, and serialized inference.
- Note that the embedded Playground works offline while the Swagger UI page loads its renderer from jsDelivr.

## Accuracy constraints

- Installation remains source-based through CMake 3.25 or newer.
- Default model download remains `scripts/download-default-model.sh`.
- Startup model precedence remains `--model`, then `PCD_GGUF`, then the default model path.
- The public API remains `GET /health`, `GET /v1/models`, `POST /v1/models/select`, and `POST /v1/pcd/decode`.
- Errors remain the stable `{ "code": "...", "message": "..." }` envelope.
- Current limits and status-code behavior must match `ui/openapi.json` and the HTTP implementation.

## Validation

Before completion:

1. Confirm every documented command, executable, endpoint, flag, and file path against the repository.
2. Confirm each referenced image exists and can be decoded.
3. Scan the README for stale claims, placeholders, malformed Markdown, and broken local links.
4. Run the existing CTest suite and report its result.
5. Keep existing unrelated working-tree changes untouched.

## Files in scope

- `README.md` — rewrite and reorganize as the product page and developer guide.
- `docs/images/*.png` — committed UI screenshots.

No server or UI behavior changes are part of this work.
