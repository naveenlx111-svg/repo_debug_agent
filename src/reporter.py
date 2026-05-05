import json
from pathlib import Path
from datetime import datetime
from src.triage import BugTarget

def generate_report(targets: list[BugTarget],repo_path: str) -> dict:
    """
    Generate a summary report from all bug targets.
    """
    fixed = [t for t in targets if t.status == "fixed"]
    failed = [t for t in targets if t.status == "failed"]
    skipped = [t for t in targets if t.status == "skipped"]

    report = {
        "repo": repo_path,
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_files_scanned": len(targets),
            "fixed": len(fixed),
            "failed": len(failed),
            "skipped": len(skipped)
        },
        "results": [
            {
                "file": t.file_path,
                "risk_score": t.risk_score,
                "suspected_issue": t.suspected_issue,
                "status": t.status,
                "explanation": t.explanation,
                "patch": t.patch,
            }
            for t in targets
        ]
    }
    return report

def save_report(report: dict) -> Path:
    """
    Save report to a JSON file with a timestamp in the name.
    Returns the path where it was saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"debug_report_{timestamp}.json")
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path

def print_report(report: dict) -> None:
    """
    Print a human readable summary to the terminal.
    """
    s = report["summary"]
    print("\n" + "="*50)
    print("REPO DEBUG AGENT — FINAL REPORT")
    print("="*50)
    print(f"Repo:      {report['repo']}")
    print(f"Time:      {report['timestamp']}")
    print(f"Scanned:   {s['total_files_scanned']} files")
    print(f"Fixed:     {s['fixed']}")
    print(f"Failed:    {s['failed']}")
    print(f"Skipped:   {s['skipped']}")
    print("="*50)

    for r in report["results"]:
        status_icon = {"fixed": "✅", "failed": "❌", "skipped": "⏭️"}.get(r["status"], "?")
        print(f"\n{status_icon} [{r['risk_score']}] {r['file']}")
        print(f"   Issue:  {r['suspected_issue']}")
        print(f"   Status: {r['status']}")
        if r["explanation"]:
            print(f"   Fix:    {r['explanation']}")