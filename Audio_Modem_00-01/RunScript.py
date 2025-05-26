import subprocess
import sys
import os

# Queue the Python files and run them one after another

base_dir = os.path.dirname(__file__)

scripts = ["Constants.py", "emitter.py", "reciever.py", "decoder.py"]

for script in scripts:
    script_path = os.path.join(base_dir, script)
    print(f"\n=== Running {script} ===")
    result = subprocess.run([sys.executable, script_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    print(result.stderr)

    if result.returncode != 0:
        print(f"{script} failed with return code {result.returncode}")
        break