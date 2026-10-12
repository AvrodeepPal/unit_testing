#unit_testing/python_test/script1.py

import subprocess

print("Script 1 running script2...")

subprocess.run(["python", "python_test/script2.py"])