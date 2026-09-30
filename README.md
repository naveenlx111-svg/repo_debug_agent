# Repo Debug Agent

An autonomous debugging agent for whole repositories. It reviews your code with an
LLM, fixes the bugs it finds, and **verifies every fix** against syntax/static checks
and your own test suite before accepting it. Your files are never touched unless you
pass `--apply`. By default you get a reviewable patch and a report.

Works with hosted models (Groq) and local ones (llama.cpp, Ollama, LM Studio, vLLM).
It's tuned to be useful with small local models like Qwen 7–9B.

```
scan → index → baseline tests → review → reproduce → fix ⟲ verify → report + patch
```

## How it works

1. **Scan.** Finds source files (Python, JS/TS, Java, Kotlin, C#, Go, Rust, C/C++, PHP, Ruby),
   honouring `.gitignore` and skipping vendored, generated, binary and oversized files.
2. **Chunk.** Splits each file into functions, methods, class headers and module-level
   code, with exact line ranges. Python uses `ast`; brace languages use a structural parser
   that isn't fooled by `{` inside strings or comments.
3. **Index.** Builds two retrieval layers:
   - a **symbol graph** (who calls what): the dependencies of the code being fixed and
     its callers, *tests first* because they document the expected behaviour;
   - a **semantic index** in ChromaDB (optional), incremental and per repo, so re-runs
     only embed chunks that changed.
4. **Baseline tests** (with `--test-cmd`). Runs your suite once in a sandbox copy of the
   repo. Failing output is fed to the reviewer and fixer as evidence.
5. **Review.** Static checks (pyflakes: undefined names, etc.) plus an LLM review of every
   non-test file. The model has to *trace* each suspicion on a concrete input before
   confirming it, which cuts false positives. Findings are mapped to the exact function
   they live in, and several findings in one function are merged into one fix.
6. **Reproduce** (Python). Before fixing, the model writes a small script that demonstrates
   the bug, and the agent runs it against the current code in the sandbox. A script that
   *passes* is evidence the report is wrong, so the issue is dismissed after one revision
   attempt. A script that fails inside the target code, or on an assertion, confirms the
   bug; its output goes into the fix prompt, and the fix has to make it pass. That gives a
   behavioural check even in repos without a test suite. Bugs that can't be scripted
   (leaks, races) fall back to the usual checks.
7. **Fix loop.** For each issue, the model gets the function, the rest of the file, related
   code from the index and any failing tests. It must answer
   `ANALYSIS → VERDICT → code → EXPLANATION`, so it can also reject a false report. The new
   function is spliced in by line range, then checked:
   - syntax and new static errors (a regression check: a checker that can't parse
     the original file doesn't count against the fix);
   - the reproduction script, which must now pass;
   - your test suite: rejected if it gets worse, and accepted fixes ratchet the baseline.

   On failure the model sees *why* (the syntax error, new undefined name, or test output)
   and tries again. It doesn't just get the same question repeated.
8. **Report.** `report.md` (human), `report.json` (machine), `fixes.patch` (`git apply`-able)
   and `transcript.jsonl` (every prompt and reply, for debugging).

## Install

```bash
git clone https://github.com/naveenlx111-svg/repo_debug_agent
cd repo_debug_agent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[rag]"          # or: pip install -r requirements.txt  (adds dev tools)
```

`[rag]` adds ChromaDB for semantic retrieval. On first use it downloads a small ONNX
embedding model (all-MiniLM-L6-v2, ~80 MB, no PyTorch needed). Without it, the agent
still runs with structural retrieval only.

## Quick start

### With a local model (llama.cpp)

```bash
llama-server -m Qwen3.5-9B-Q4_K_M.gguf --port 8080 --ctx-size 16384 --gpu-layers all
repo-debug-agent path/to/repo --provider local --test-cmd "pytest -q"
```

The model name is auto-detected from the server. If the server was started with
`--api-key`, put that key in `LLM_API_KEY`. Thinking is switched off for hybrid
reasoning models (Qwen3.x): the prompts already make the model reason before deciding,
and thinking made runs several times slower.

### With Groq (hosted, free tier)

```bash
echo "GROQ_API_KEY=your_key" > .env
repo-debug-agent path/to/repo --test-cmd "pytest -q"
```

Uses `llama-3.1-8b-instant` for review and `llama-3.3-70b-versatile` for fixes.

### With Ollama

```bash
repo-debug-agent path/to/repo --provider ollama --model qwen2.5-coder:7b
```

## Usage

```bash
repo-debug-agent REPO                         # review + fix; writes a patch, repo untouched
repo-debug-agent REPO --test-cmd "pytest -q"  # verify each fix with your tests (recommended)
repo-debug-agent REPO --analyze-only          # just list suspected bugs
repo-debug-agent REPO --apply                 # write accepted fixes into the repo
git -C REPO apply debug_reports/<run>/fixes.patch   # ...or apply the patch yourself later
```

`python main.py REPO ...` also works from a source checkout without installing.

| Option | Default | |
| --- | --- | --- |
| `--provider {groq,local,ollama}` | groq if `GROQ_API_KEY` is set, else local | |
| `--base-url`, `--model`, `--triage-model`, `--fix-model` | from provider | any OpenAI-compatible endpoint |
| `--test-cmd CMD` | none | run in a sandbox copy; strongly recommended |
| `--apply` | off | write fixes into the repo (refuses files you edited during the run) |
| `--analyze-only` | off | report issues without fixing |
| `--min-confidence F` | 0.6 | skip findings the reviewer is less sure about |
| `--max-issues N` | 20 | cap on fixes per run |
| `--max-attempts N` | 3 | fix attempts per issue, each with feedback |
| `--no-repro` | off | skip reproduce-before-fix (it runs model-written scripts) |
| `--python PATH` | repo's `.venv`, else `python3` | interpreter for reproduction scripts |
| `--include-tests` | off | also review/fix test files (off: tests are the spec) |
| `--no-embeddings` | off | structural retrieval only |
| `--embed-model NAME` | `default` | or any sentence-transformers model (`pip install -e ".[st]"`) |
| `--reindex` | off | rebuild the semantic index from scratch |
| `--workers N` | 4 (groq), 1 (local) | parallel file reviews; raise it only if your local server has several slots (`--parallel`) |
| `--out DIR` | `./debug_reports` | reports, patch and transcript go in a timestamped subfolder |
| `-v` | off | stage timings and more detail |

Environment variables (`.env` is loaded from the current directory): `GROQ_API_KEY`,
`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_TRIAGE_MODEL`,
`LLM_FIX_MODEL`, `LLM_TIMEOUT`, `LLM_EXTRA_BODY` (JSON merged into every request).
See `.env.example`.

## Safety model

- Edits happen in memory. With `--test-cmd`, in a temporary sandbox copy of the repo
  (`node_modules`, `.venv` and similar are symlinked, not copied, and `PYTHONPATH` points
  at the sandbox so tests import the edited code, not an installed copy).
- Reproduction runs **model-written scripts**. They run in the sandbox copy, with a
  30-second timeout, and the sandbox copies of source files are restored after every
  run. They are not otherwise isolated (no container), so use `--no-repro` for
  code you wouldn't let a script touch.
- Test files are not modified by default, so the agent can't make tests pass by
  weakening them.
- `--apply` writes only files whose on-disk content still matches what the run started
  from. Line endings (LF/CRLF) are preserved.
- Every accepted fix records how it was verified: `tests` (the suite improved),
  `repro` (the reproduction script failed before and passes after), `no-regressions`
  (tests ran, but none covered the bug), `syntax`, or `none`
  (no checker for that language). **Review anything not verified by tests.**

## Results on the sample repo

`examples/sample_repo` has 8 realistic planted bugs (no hint comments), some correct
code the agent should leave alone, and a test suite that pins the right behaviour
(8 failing tests). With **Qwen3.5-9B Q4_K_M** on an 8 GB laptop GPU:

```bash
repo-debug-agent examples/sample_repo --provider local --test-cmd "python -m pytest -q"
```

| | |
| --- | --- |
| Bugs found | 7 / 8 (missed: the unclosed file in `word_count`) |
| False positives | 0 |
| Fixed | 7 / 7, each on the first attempt, all minimal one- or two-line changes |
| Tests | 8 failing → all passing |
| Time | ~2 minutes, 11 LLM calls |

## Limitations

- **Small models over-report on real code.** On production-quality code a 7–9B model
  flags plausible-sounding non-bugs. The reviewer's self-check and the fixer's verdict
  step catch many of these, but not all. A test suite is what makes the output
  trustworthy. Without one, treat the patch as suggestions, and consider a larger
  `--fix-model`.
- Fixes rewrite one function (or class, or block of module code) at a time. Bugs
  that need coordinated edits across several functions or files are out of scope, and
  functions over 300 lines are skipped.
- Chunking for non-Python languages is structural, not a full parser. It handles
  common styles well but can mis-split unusual code (JS regex literals containing
  braces, macros).
- Syntax verification needs the language's tool on `PATH` (`node`, `gofmt`, `javac`,
  `gcc`/`g++`, `ruby`, `php`). Python is always checked.

## Project layout

```
src/repo_debug_agent/
  cli.py          argument parsing, logging, exit codes
  config.py       settings + provider presets (no env/network access at import time)
  pipeline.py     orchestration of the stages
  crawler.py      file discovery (.gitignore-aware)
  languages.py    language registry
  chunker.py      function/class/module chunks with exact line ranges
  retrieval/      symbol graph + ChromaDB semantic index + context assembly
  analyzer.py     static checks + LLM review, issue localization and merging
  repro.py        reproduce-before-fix: model-written failing scripts, run in the sandbox
  fixer.py        the propose → splice → check → reproduce/test → retry loop
  validation.py   syntax/static checks, test runner, failure-count comparison
  workspace.py    in-memory overlay, sandbox, diff, conflict-safe apply
  llm.py          OpenAI-compatible client, reply parsing, transcript recording
  prompts.py      every prompt, in one place
  report.py, ui.py
tests/            offline test suite (a scripted fake LLM drives full pipeline runs)
examples/sample_repo/
```

## Development

```bash
pip install -e ".[rag,dev]"
pytest            # ~110 tests, offline, a few seconds
ruff check src tests && ruff format src tests
```
