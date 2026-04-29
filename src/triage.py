# src/triage.py
import json
from pathlib import Path
from dataclasses import dataclass
from groq import Groq

from src.config import (
    GROQ_API_KEY,
    TRIAGE_MODEL,
    MAX_FILE_PREVIEW,
    MIN_RISK_SCORE
)

client = Groq(api_key=GROQ_API_KEY)

@dataclass
class BugTarget:
    """Represents a file flagged by triage as worth investigating."""
    file_path:       str
    risk_score:      float
    suspected_issue: str
    status:          str = "pending"
    patch:           str = None
    explanation:     str = None

def triage_file(path: Path) -> BugTarget | None:
    try:
        code = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None

    preview = code[:MAX_FILE_PREVIEW]

    prompt = (
        "You are a code reviewer. Analyze this code file for bugs.\n"
        "Respond ONLY with a JSON object, no explanation outside the JSON:\n"
        "{\n"
        '  "risk_score": <float 0-10, where 10 means definite bugs>,\n'
        '  "suspected_issue": "<one sentence describing the most likely bug or none>",\n'
        '  "worth_investigating": <true or false>\n'
        "}\n\n"
        f"File: {path.name}\n"
        f"{preview}"
    )

    try:
        response = client.chat.completions.create(
            model       = TRIAGE_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
            max_tokens  = 200,
        )

        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        if (
            result.get("worth_investigating") and
            result.get("risk_score", 0) >= MIN_RISK_SCORE
        ):
            return BugTarget(
                file_path       = str(path),
                risk_score      = result["risk_score"],
                suspected_issue = result["suspected_issue"]
            )
        return None

    except json.JSONDecodeError:
        print(f"  Warning: LLM returned invalid JSON for {path.name}")
        return None
    except Exception as e:
        print(f"  Warning: triage failed for {path.name}: {e}")
        return None

def triage_repo(files: list[Path]) -> list[BugTarget]:
    print(f"\nTriaging {len(files)} files...")
    targets = []

    for path in files:
        print(f"  Checking {path.name}...", end=" ", flush=True)
        target = triage_file(path)
        if target:
            print(f"risk={target.risk_score}  {target.suspected_issue}")
            targets.append(target)
        else:
            print("clean")

    targets.sort(key=lambda t: t.risk_score, reverse=True)
    print(f"\nFound {len(targets)} files worth investigating.")
    return targets