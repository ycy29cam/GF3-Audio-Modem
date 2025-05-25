import subprocess

if __name__ == "__main__": # used to skip emitter-reciever to work on prerecorded data
    subprocess.run("Constants.py", shell=True)
    # subprocess.run("emitter.py", shell=True)
    # subprocess.run("reciever.py", shell=True)
    subprocess.run("decoder.py", shell = True)