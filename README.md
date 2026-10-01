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
5. **Review.** Static checks (pyflakes: undefined names, invalid escapes, etc.) plus an LLM
   review of every non-test file. The model has to *trace* each suspicion on a concrete
   input before confirming it, which cuts false positives. Findings are mapped to the exact
   function they live in, and several findings in one function are merged into one fix.
   Where the server supports it (llama.cpp does), output is constrained to a JSON schema,
   so an unescaped quote in a small model's reasoning can't derail the reply; this took
   well-formed replies on a problem prompt from 6/8 to 8/8, and made them 2.7x faster.
   If a model still loops into the token limit, the complete findings it produced first
   are kept. Reviews are cached, so re-runs only re-review changed code.
6. **Reproduce** (Python). Before fixing, the model writes a small script that demonstrates
   the bug, using only inputs the real callers can produce (it is shown how the code is
   called) and no mocks (scripts that use them are rejected). The agent runs it against the
   current code in the sandbox:
   - the script **passes**: evidence the report is wrong, so after one revision attempt the
     issue is dismissed;
   - it **fails** inside the target code or on an assertion: its output goes into the fix
     prompt, and a fix is accepted only once the script passes;
   - it **can't run**, or the bug can't be scripted (leaks, races): the usual checks apply.

   A failing script is then checked by a separate, skeptical call that sees the real callers:
   could the program actually pass that input, and does the assertion match the intended
   behaviour? A script that fails the check gets one rewrite; if the rewrite also fails it,
   the reproduction doesn't count as evidence. The bug isn't dismissed, because the check
   can be wrong too.
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
   **No evidence, no edit.** A fix is only *applied* (kept for later fixes, written to
   `fixes.patch`, used by `--apply`) when something shows the bug was real: the test suite
   improved, a checked reproduction script failed before and passes after, or a static
   finding (e.g. an undefined name) went away. A fix that only passes syntax checks is a
   **suggestion**: it's in the report and `suggestions.patch`, but never applied.
   `--keep-unverified` turns suggestions back into fixes.
8. **Report.** `report.md` (human), `report.json` (machine), `fixes.patch` (verified fixes,
   `git apply`-able), `suggestions.patch` (unverified)
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
| `--apply` | off | write verified fixes into the repo (refuses files you edited during the run) |
| `--keep-unverified` | off | treat syntax-checked-only fixes as fixes, not suggestions |
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
| `--no-cache` | off | re-review every file; by default, reviews of unchanged code are reused |
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
- Every fix records how it was verified. `tests` (the suite improved), `repro` (a checked
  reproduction script failed before and passes after) and `static` (the static finding is
  gone) count as evidence; `no-regressions` (tests ran, none covered the bug), `syntax` and
  `none` don't, so those fixes are suggestions. **Review anything not verified by tests.**

## Results

All numbers below are from **Qwen3.5-9B Q4_K_M** on an 8 GB laptop GPU (~38 tokens/s).

### Sample repo (planted bugs, with tests)

`examples/sample_repo` has 8 realistic planted bugs (no hint comments), some correct
code the agent should leave alone, and a test suite that pins the right behaviour
(8 failing tests).

```bash
repo-debug-agent examples/sample_repo --provider local --test-cmd "python -m pytest -q"
```

| | |
| --- | --- |
| Bugs found | 7 / 8 (missed: the unclosed file in `word_count`) |
| False reports | 0 |
| Applied fixes | 6, each on the first attempt, each reproduced by a script *and* verified by the test suite |
| Suggestions | 1: `cartTotal` (JavaScript) is a real bug, but no test covers it and reproduction is Python-only, so it isn't applied |
| Tests | 8 failing → all passing |
| Time | 3.3 min, 23 LLM calls; a re-run reuses every review from cache and takes 2 min |

### Real code, no tests (the hard case)

The agent's own source (22 files), with no test command, so reproduction and static
checks are the only evidence. Three runs from the *same* cached findings (11 selected):
without reproduction, with it, and with the current evidence policy (fixes without
evidence become suggestions; reproduction scripts are checked against the real callers):

| | no reproduction | reproduction | **+ evidence policy** |
| --- | --- | --- | --- |
| Real bug (empty `LLM_TIMEOUT` rejected) fixed | yes, syntax-checked only | yes, script-verified | yes, script-verified |
| **Unneeded edits in the applied patch** | 5 | 4 | **0** |
| Edits that broke correct logic, applied | 1 | 0 | 0 |
| Unverified fixes held back as suggestions | – | – | 3 |
| False reports dismissed | 2 | 5 | 5 |

So on code without tests, the patch now contains only what was shown to be a real bug;
the rest is clearly marked as unverified suggestions.

A further run of the complete current system (schema-constrained reviews, fresh findings)
on the same code: 17 suspected issues, 13 dismissed (7 because the reproduction script
passed on the current code), 3 unverified suggestions, **0 applied edits**. One of the
suggestions would have changed correct logic, which the old behaviour would have applied.
That run also shows the remaining weak spot: the reviewer didn't flag the real
`LLM_TIMEOUT` bug at all (it made a different, invalid claim about the same function).
Which bugs the review finds still varies from run to run with a small model.

## Limitations

- **Small models over-report on real code, and their findings vary.** On production-quality
  code a 7–9B model flags plausible-sounding non-bugs. Most are dismissed, and the rest end
  up as suggestions rather than applied edits, but the suggestions list still needs a human
  look. Which real bugs get flagged also varies between runs; a larger `--triage-model` or
  a second run helps. A test suite is what turns real bugs into verified fixes; without one,
  only reproducible Python bugs and static findings can be verified.
- A reproduction script is the model's claim, not proof. It can use an input the real
  callers never produce, or assert the wrong expectation. The script is in the report
  so you can check it quickly.
- Fixes rewrite one function (or class, or block of module code) at a time. Bugs
  that need coordinated edits across several functions or files are out of scope, and
  functions over 300 lines are skipped.
- Chunking for non-Python languages is structural, not a full parser. It handles
  common styles well but can mis-split unusual code (JS regex literals containing
  braces, macros). Reproduction is Python-only.
- Syntax verification needs the language's tool on `PATH` (`node`, `gofmt`, `javac`,
  `gcc`/`g++`, `ruby`, `php`). Python is always checked.

## Project layout

```
src/repo_debug_agent/
  cli.py          argument parsing, logging, exit codes
  cache.py        review reply cache (re-runs skip unchanged code)
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
pytest            # ~140 tests, offline, a few seconds
ruff check src tests && ruff format src tests
```
