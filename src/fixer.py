from pathlib import Path
from groq import Groq
import chromadb

from src.config import (
    GROQ_API_KEY,
    FIX_MODEL,
    MAX_RETRIES
)

from src.triage import BugTarget
from src.retriever import retrieve,format_context

client = Groq(api_key = GROQ_API_KEY)

def fix_bug(target: BugTarget, collection: chromadb.Collection) -> BugTarget:
    """
    The core agent loop for one bug target.
    Retrieves context, asks LLM to fix, applies fix, verifies.
    """
    print(f"\n Fixing: {target.file_path}")
    print(f"  Suspected: {target.suspected_issue}")

    original_code = Path(target.file_path).read_text(encoding='utf-8',errors='replace')

    for attempt in range(1,MAX_RETRIES+1):
        print(f"\nAttempt {attempt} to fix the bug...")

        rag_chunks = retrieve(
            query = f"bug: {target.suspected_issue} in {Path(target.file_path).name}",
            collection = collection,
            n =5
        )
        context = format_context(rag_chunks)

        prompt = (
            "You are an expert Python debugger.\n"
            "You will be given a buggy file and related context.\n\n"
            "Your job:\n"
            "1. Find the buggy function\n"
            "2. Rewrite ONLY that function with the fix applied\n"
            "3. Return your response in this EXACT format:\n\n"
            "BUGGY_FUNCTION:\n"
            "<paste the exact original buggy function here>\n\n"
            "FIXED_FUNCTION:\n"
            "<paste the complete fixed function here>\n\n"
            "EXPLANATION:\n"
            "<one sentence explaining what you fixed>\n\n"
            f"Suspected issue: {target.suspected_issue}\n\n"
            f"=== BUGGY FILE: {target.file_path} ===\n"
            f"{original_code}\n\n"
            f"=== RELATED CONTEXT FROM CODEBASE (via RAG) ===\n"
            f"{context}"
        )

        try:
            response = client.chat.completions.create(
                model = FIX_MODEL,
                messages = [{"role": "user", "content": prompt}],
                temperature = 0.1,
                max_tokens = 1000,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"    LLM call failed: {e}")
            continue

        parsed = parse_fix_response(raw)
        if not parsed:
            print(f"    Could not parse LLM response, retrying...")
            continue
        buggy_fn, fixed_fn, explanation = parsed

        if buggy_fn not in original_code:
            print(f"    Could not find buggy function in file, retrying...")
            continue

        new_code = original_code.replace(buggy_fn, fixed_fn)
        Path(target.file_path).write_text(new_code,encoding='utf-8')
        print(f"    Fix applied!")

        if verify_syntax(target.file_path):
            target.status = "fixed"
            target.patch = f"BEFORE:\n{buggy_fn}\n\nAFTER:\n{fixed_fn}"
            target.explanation = explanation
            print(f"    Syntax OK! Fix successful.")
            return target
        else:
            print(f"    Syntax error after fix — restoring original...")
            Path(target.file_path).write_text(original_code,encoding='utf-8')
            continue
    
    target.status = "failed"
    target.explanation = f"Could not fix after {MAX_RETRIES} attempts."
    return target


def parse_fix_response(raw: str) -> tuple | None:
    """
    Parse LLM response into (buggy_function,fixed_function,explanation).
    Returns None if parsing fails.
    """
    try:
        parts = raw.split("BUGGY_FUNCTION:")
        if len(parts) < 2:
            return None
        
        rest = parts[1]
        parts2 = rest.split("FIXED_FUNCTION:")
        if len(parts2) < 2:
            return None
        
        buggy_fn = parts2[0].strip()

        parts3 = parts2[1].split("EXPLANATION:")
        fixed_fn = parts3[0].strip()
        explanation = parts3[1].strip() if len(parts3) > 1 else "No explanation provided."

        # strip markdown fences if present
        for marker in ["```python","```"]:
            ibuggy_fn = buggy_fn.replace(marker,"").strip()
            fixed_fn = fixed_fn.replace(marker,"").strip()

        return buggy_fn,fixed_fn,explanation
    except Exception:
        return None
    

def verify_syntax(file_path: str) -> bool:
    """
    Check if a file has valid syntax after patching.
    Handles multiple languages.
    """
    path = Path(file_path)
    ext  = path.suffix.lower()

    try:
        code = path.read_text(encoding="utf-8", errors="replace")

        # ── Python ────────────────────────────────────────────────────
        if ext == ".py":
            compile(code, file_path, "exec")
            return True

        # ── JavaScript / TypeScript ───────────────────────────────────
        elif ext in (".js", ".ts"):
            import subprocess
            result = subprocess.run(
                ["node", "--check", str(path)],
                capture_output=True, text=True
            )
            return result.returncode == 0

        # ── Java ──────────────────────────────────────────────────────
        elif ext == ".java":
            import subprocess
            result = subprocess.run(
                ["javac", str(path)],
                capture_output=True, text=True
            )
            return result.returncode == 0

        # ── Go ────────────────────────────────────────────────────────
        elif ext == ".go":
            import subprocess
            result = subprocess.run(
                ["go", "vet", str(path)],
                capture_output=True, text=True
            )
            return result.returncode == 0

        # ── Unknown language — skip syntax check ─────────────────────
        else:
            print(f"    No syntax checker for {ext} — assuming OK")
            return True

    except SyntaxError as e:
        print(f"    SyntaxError: {e}")
        return False
    except FileNotFoundError:
        # node/javac/go not installed — skip check
        print(f"    Syntax checker not found for {ext} — assuming OK")
        return True
    except Exception as e:
        print(f"    Syntax check failed: {e}")
        return True  # don't block the fix if checker crashes