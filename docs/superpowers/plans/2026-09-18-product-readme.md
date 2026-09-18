# PCD Server Product README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use Markdown checkboxes for tracking.

**Goal:** Replace the repository README with a polished, product-first page that gets developers from discovery to a successful native REST request and illustrates the embedded web UI.

**Architecture:** Keep the documentation self-contained in `README.md`, organized through progressive disclosure from value proposition to quick start, API use, operations, and internals. Promote clean existing browser captures from the ignored Playwright output into a stable `docs/images/` directory; do not change application behavior.

**Tech Stack:** GitHub Flavored Markdown, PNG browser captures, CMake 3.25+, C++20, curl, CTest, OpenAPI 3.1

---

## File structure

- `README.md` — product landing page, native setup guide, REST API guide, operational reference, and architecture explanation.
- `docs/images/pcd-playground.png` — primary successful-decode screenshot.
- `docs/images/pcd-scenarios.png` — scenario-driven multi-field extraction screenshot.
- `docs/images/pcd-tetris.png` — Tetris PCD visualization screenshot.
- `docs/images/pcd-api-docs.png` — rendered OpenAPI/Swagger UI screenshot.

### Task 1: Promote stable UI screenshots

**Files:**
- Create: `docs/images/pcd-playground.png`
- Create: `docs/images/pcd-scenarios.png`
- Create: `docs/images/pcd-tetris.png`
- Create: `docs/images/pcd-api-docs.png`

- [x] **Step 1: Create the image directory**

Run:

```bash
mkdir -p docs/images
```

Expected: `docs/images/` exists without modifying ignored Playwright output.

- [x] **Step 2: Copy the selected successful-state captures**

Run:

```bash
cp .playwright-mcp/ui-1.png docs/images/pcd-playground.png
cp .playwright-mcp/ui-preset.png docs/images/pcd-scenarios.png
cp .playwright-mcp/ui-tetris-end.png docs/images/pcd-tetris.png
cp .playwright-mcp/ui-docs.png docs/images/pcd-api-docs.png
```

Expected: four PNG files exist under `docs/images/`; the Tetris image shows a completed game rather than a transient request error.

- [x] **Step 3: Validate the assets**

Run:

```bash
file docs/images/*.png
```

Expected: all four files report `PNG image data` with non-zero dimensions.

### Task 2: Rewrite the README as a product page

**Files:**
- Modify: `README.md`

- [x] **Step 1: Replace the opening with the product hero**

Use the title `PCD Server`, a direct outcome-oriented subtitle, restrained platform/API badges, navigation links, and the main image:

```markdown
# PCD Server

**Turn text into guaranteed bounded JSON with a local GGUF model.**

PCD Server scores only the values your application allows and decides every field in a few batched forward passes. You get valid JSON, a probability distribution for every decision, and no generated syntax to parse or repair.

![PCD Server Playground showing a completed fraud-triage decode](docs/images/pcd-playground.png)
```

Immediately follow with four product benefits: constrained output, parallel field decisions, local inference, and reusable schema checkpoints.

- [x] **Step 2: Add the native quick start and first request**

Document prerequisites for Apple Silicon macOS and Linux, then provide this sequential path:

```bash
git clone https://github.com/stephanj/pcdServer.git
cd pcdServer
./scripts/download-default-model.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/pcd_server
```

The clone URL comes from the configured `origin` remote. Follow the commands with `GET /health`, the embedded Playground/docs URLs, and a complete inline `POST /v1/pcd/decode` request using priority and fraudulent fields. Include a compact representative response that shows `values`, per-field probabilities, and metrics.

- [x] **Step 3: Add the UI tour**

Reference all four committed assets with meaningful alt text. Use the scenario, Tetris, and API-doc captures in compact subsections explaining what each interface demonstrates. State that the server embeds the UI, that `/openapi.json` works offline, and that `/docs` obtains Swagger UI assets from jsDelivr.

- [x] **Step 4: Build the API and operations reference**

Include:

- an endpoint table for `GET /health`, `GET /v1/models`, `POST /v1/models/select`, and `POST /v1/pcd/decode`;
- the stable error envelope and status-code table;
- input limits from `ui/openapi.json`;
- model discovery and transactional switching behavior;
- the complete server flag table and startup precedence;
- the schema cache status meanings and response field semantics;
- concurrency, native test, end-to-end test, and benchmark commands.

- [x] **Step 5: Retain the technical depth below the onboarding path**

Move the seven-stage decode explanation and full-checkpoint rationale below the API guide. Finish with an explicit current-scope section: booleans and bounded string enums are supported; free strings, numbers, arrays, arbitrary JSON Schema, authentication, and multi-model residency are not.

- [x] **Step 6: Check prose and Markdown locally**

Run:

```bash
grep -nE 'T[B]D|T[O]DO|F[I]XME|PLACEH[O]LDER' README.md
git diff --check -- README.md
```

Expected: `grep` returns no matches and `git diff --check` returns no errors.

### Task 3: Verify documentation against the product

**Files:**
- Verify: `README.md`
- Verify: `docs/images/*.png`
- Reference: `src/main.cpp`
- Reference: `ui/openapi.json`
- Reference: `scripts/download-default-model.sh`

- [x] **Step 1: Check documented local links and assets**

Run a script that extracts repository-relative Markdown link targets from `README.md`, ignores anchors and HTTP URLs, removes optional `#fragment` suffixes, and asserts each target exists.

Expected: every local link and image resolves.

- [x] **Step 2: Cross-check command-line and API facts**

Run:

```bash
./build/pcd_server --help
python3 -m json.tool ui/openapi.json >/dev/null
```

Expected: documented flags match help output and the OpenAPI document is valid JSON.

- [x] **Step 3: Run the existing test suite**

Run:

```bash
ctest --test-dir build --output-on-failure
```

Expected: all configured tests pass.

- [x] **Step 4: Review the final diff without disturbing unrelated work**

Run:

```bash
git diff --check
git diff -- README.md docs/images
git status -sb
```

Expected: no whitespace errors; the diff contains only the intended README and image additions plus the user's pre-existing modifications to `ui/index.html` and `ui/tetris.js` in status.

- [x] **Step 5: Commit only the completed documentation**

Run:

```bash
git add README.md docs/images docs/superpowers/plans/2026-09-18-product-readme.md
git commit -m "docs: create PCD Server product README"
```

Expected: the README, screenshots, and implementation plan are committed; `ui/index.html` and `ui/tetris.js` remain uncommitted and untouched.
