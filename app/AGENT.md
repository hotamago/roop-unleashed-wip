# AGENT.md

This file defines coding and execution rules for agents and contributors working inside `./app`.

## 1. Goals

- Write code that is easy to read, debug, and extend.
- Do not hide failures with silent caps or fallback hacks unless the root cause is understood.
- Optimize for heavy workloads: video, batch processing, CPU/GPU pipelines, large IO, and memory pressure.
- Prefer small, explicit, verifiable changes.

## 2. Environment and test execution

This repo uses `uv`. When installing dependencies or running tests, use `uv` so the environment matches the lockfile.

Run from `./app`:

```powershell
cd app
uv sync --dev
uv run pytest
```

Run one test file:

```powershell
cd app
uv run pytest tests/test_settings_and_memory.py
```

Run a filtered test selection:

```powershell
cd app
uv run pytest tests/test_one_chain_executor.py -k worker
```

Quick runtime dependency check:

```powershell
cd app
uv sync --dev
uv run python -c "import insightface, onnxruntime, torch; print('ok')"
```

Rules:

- Use `uv run pytest ...`, not ad-hoc `pytest` from whatever interpreter happens to be active.
- When fixing a bug, run the nearest relevant tests first, then expand coverage.
- If tests are blocked by missing packages, drivers, or runtime dependencies, report the blocker clearly instead of guessing.

## 3. Code standards

- Prefer simple, explicit code over clever code.
- Function and method names must describe real behavior.
- Each function should have one clear responsibility.
- Keep business logic out of UI callbacks when it can live in a dedicated module.
- Do not introduce abstraction early for a single use case.
- Do not change user-requested semantics with silent fallback behavior.
- If a user requests `N` workers, `N` batches, or `N` threads, the system should try to run `N`. If it fails, expose the failure clearly so the real issue can be fixed.

## 4. Design for extensibility

- Prefer adding a small function or focused module over growing giant if/else chains.
- Important runtime limits must flow through one clear config or resolver path.
- Runtime config should move through a clear chain: `Settings` -> planner/resolver -> executor.
- Keep these concerns separate:
  - config parsing
  - resource planning
  - scheduling
  - processing
  - UI presentation
- UI text must match actual runtime behavior.
- If a data contract changes, update:
  - call sites
  - tests
  - status or message text
  - any short docs that describe the behavior

## 5. Professional code style

- Keep functions short enough to read in one screen when practical.
- Avoid deep nesting; prefer early returns.
- Add comments only when they explain why, not when they restate the code.
- Use type hints on important internal APIs.
- If a return value carries multiple meanings, prefer a clearly named object or dict over a hard-to-read tuple, unless the tuple pattern is already small and stable.
- Avoid vague variable names like `data`, `tmp`, or `ret2` when a clearer name is possible.
- Make side effects obvious. If a function mutates state, that should be easy to tell.

## 6. Bug-fix rules

- Find the root cause before adding caps, retries, fallbacks, or clamps.
- Do not silently lower workers, threads, or batch sizes just to create fake stability.
- If a temporary guard is necessary, document:
  - why it exists
  - when it triggers
  - what root cause still needs to be fixed
- Error messages must be actionable. They should identify the parameter, stage, model, or worker involved.

## 7. Heavy-task optimization rules

This repo processes heavy media workloads. Every change should consider:

- throughput
- peak RAM and VRAM
- queue backpressure
- serialization cost
- `numpy` frame copy cost
- thread contention
- model or session initialization cost

Rules:

- Do not copy frames unless needed.
- Reuse processor instances, sessions, and worker pools when it is safe.
- Avoid repeated model load/unload in hot paths.
- Move warm-up and initialization outside frame loops.
- Long tasks must expose progress and status so bottlenecks are visible.
- Prefer bounded queues over unbounded queues.
- For large caches, write by chunk or segment instead of only at the end.

## 8. CPU multi-worker rules

For heavy CPU work, use multi-worker execution instead of forcing everything through one thread.

Prefer:

- `ProcessPoolExecutor` or dedicated worker processes for truly CPU-bound Python or numpy work when the GIL is a real bottleneck
- `ThreadPoolExecutor` for IO-bound work or runtimes that release the GIL

Implementation rules:

- Worker count must come from a clear config or resolved runtime value.
- Do not silently cap worker count without a very strong technical reason and clear documentation.
- Workers should receive the minimum required data.
- Do not serialize large objects across processes if rebuilding them inside the worker is cheaper.
- Chunk work large enough to reduce dispatch overhead, but not so large that memory spikes.
- Keep output ordering deterministic when frame or video order matters.
- Shutdown must be clean: release worker resources, close queues, and flush writers.

For CPU-heavy pipelines:

- Prefer chunk-level parallelism over a huge number of tiny per-frame tasks when overhead is high.
- Benchmark before claiming an optimization.
- If a GPU stage already exists, do not let CPU producers outrun downstream consumers without backpressure.

## 9. GPU and mixed workload rules

- GPU workers and sessions are expensive, but do not hide GPU failures with arbitrary hard caps.
- If GPU concurrency breaks, investigate the real cause:
  - session or thread safety
  - allocator pressure
  - model initialization per worker
  - buffer copy overhead
  - stream contention
- Separate CPU-bound and GPU-bound stages clearly.
- Avoid unnecessary global locks that stall the whole pipeline.
- The memory planner should describe and resolve resources clearly, not silently override user intent.

## 10. Test strategy

When changing code:

- Add or update tests close to the changed behavior.
- Tests should describe behavior, not just current implementation details.
- For concurrency and resource bugs, test:
  - requested worker count
  - ordering
  - worker or session reuse
  - failure propagation
  - cleanup paths
- For planner and config changes, test:
  - config values flow all the way into executors
  - UI or status text does not misrepresent runtime behavior

Minimum completion checklist:

- Run `uv run pytest` for the affected area when the environment allows it.
- If tests cannot run, state exactly what dependency or runtime piece is missing.
- Use `python -m py_compile` on changed files when a quick syntax check is useful.

## 11. Anti-patterns

- Do not add hard caps just because crashes are inconvenient.
- Do not swallow exceptions and return fake-success fallbacks.
- Do not change UI labels without changing runtime behavior.
- Do not over-abstract early.
- Do not add new global mutable state if dependency injection through context or objects is practical.
- Do not use "safe defaults" that change what the user explicitly asked for.

## 12. Definition of done

A change is considered good when:

- A reader can quickly understand what it does.
- Runtime behavior matches user-provided config.
- Failures expose useful clues about the real issue.
- There is a test or at least a clear verification path.
- It does not make heavy pipelines worse in throughput or memory use without a documented reason.
