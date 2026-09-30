"""Run the agent from a source checkout without installing it: python main.py PATH [options].

Installed usage: `pip install -e .` then `repo-debug-agent PATH`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from repo_debug_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
