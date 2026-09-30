import io

from rich.console import Console

from repo_debug_agent.models import FixResult, FixStatus, Issue
from repo_debug_agent.ui import UI


def test_model_text_is_not_parsed_as_rich_markup():
    out = io.StringIO()
    ui = UI(Console(file=out, width=200, highlight=False))
    issue = Issue("stats.py", "uses ordered[mid + 1]", line=15, symbol="median", id="B1")
    ui.issues([issue], dropped=0)
    ui.fix_started(issue)
    ui.fix_finished(
        FixResult(issue, FixStatus.FIXED, "Use ordered[mid - 1] and [bold]ordered[mid][/]")
    )
    ui.warn("path/[weird]/file.py")
    text = out.getvalue()
    assert "ordered[mid + 1]" in text
    assert "Use ordered[mid - 1] and [bold]ordered[mid][/]" in text
    assert "path/[weird]/file.py" in text
