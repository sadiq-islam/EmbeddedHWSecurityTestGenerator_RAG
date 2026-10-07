"""Create the environment once and open the browser application."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv

def main():
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = environment / ".requirements-hash"

    fingerprint = hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest()

    if sys.version_info < (3, 11):
        raise RuntimeError("Install Python 3.11 or newer")

    if not python.exists():
        print("Preparing Python environment…", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)

    if not marker.exists() or marker.read_text() != fingerprint:
        print("Installing dependencies; the first run can take several minutes…", flush=True)
        subprocess.run([str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")], check=True)
        marker.write_text(fingerprint)

    return subprocess.call([str(python), "-m", "streamlit", "run", str(root / "app.py"),
                            "--server.address", "127.0.0.1", "--browser.gatherUsageStats", "false"], cwd=root)

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)