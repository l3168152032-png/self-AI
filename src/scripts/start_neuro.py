import os
_dll_dir = os.environ.get("NEURO_DLL_DIR")
if _dll_dir and os.path.exists(_dll_dir):
    os.add_dll_directory(_dll_dir)
import subprocess
import time
import sys

def start_neuro():
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    body_script = os.path.join(REPO_ROOT, "src", "body", "neuro_body.py")
    brain_script = os.path.join(REPO_ROOT, "src", "core", "neuro_brain.py")
    print("[launch] starting Neuro system...")

    print("[launch] starting body (VTS)...")
    body_process = subprocess.Popen([sys.executable, body_script], cwd=REPO_ROOT)
    time.sleep(2)

    print("[launch] starting brain (LLM)...")
    try:
        subprocess.run([sys.executable, brain_script], cwd=REPO_ROOT)
    except KeyboardInterrupt:
        print("\n[launch] shutting down...")
    finally:
        body_process.terminate()
        print("[launch] done")

if __name__ == "__main__":
    start_neuro()