import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "Constants.py"])
    subprocess.run([sys.executable, "emitter.py"])
    # subprocess.run([sys.executable, "reciever.py"])
    # subprocess.run([sys.executable, "decoder.py"])