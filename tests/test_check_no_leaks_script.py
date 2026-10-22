import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    return subprocess.run([sys.executable, str(REPO_ROOT / "scripts/check_no_leaks.py"), *args], cwd=REPO_ROOT,
                          capture_output=True, text=True)


def test_the_script_always_prints_a_summary_line():
    result = _run()
    assert "leak check:" in result.stdout.splitlines()[-1]


def test_a_missing_extra_file_fails_instead_of_passing_silently():
    result = _run("--extra", "data/this_file_does_not_exist.md")
    assert result.returncode == 1
    assert "requested with --extra but not found" in result.stdout
    assert result.stdout.splitlines()[-1].startswith("leak check:")
