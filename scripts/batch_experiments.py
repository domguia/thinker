import torch
import os
import subprocess

# Define texts
texts = {
    "short": "The Moon is Earth's only natural satellite.",
    "medium": "The Moon is Earth's only natural satellite. It is the fifth largest satellite in the Solar System and the largest and most massive relative to its parent planet.",
    "long": "The Moon is Earth's only natural satellite. It is the fifth largest satellite in the Solar System and the largest and most massive relative to its parent planet. At a mean distance of 384,400 km, it is about 30 times the diameter of Earth. The Moon's surface is covered in impact craters and vast, dark volcanic plains called maria. It is in synchronous rotation with Earth, meaning it always shows the same face. The Moon's gravitational pull is the primary driver of Earth's tides. Its presence stabilizes Earth's axial tilt, which is crucial for a stable climate. Although it appears bright in the night sky, its surface is actually dark, with a reflectance slightly higher than that of worn asphalt."
}

prompt_lengths = [5, 10, 20]
exp_name = "wiki_length_study"

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(result.stdout)

# Ensure PYTHONPATH is set
os.environ["PYTHONPATH"] = os.getcwd()

for text_key, text in texts.items():
    for n_prompt in prompt_lengths:
        run_id = f"{text_key}_p{n_prompt}"
        cmd = (
            f"conda run -n thinker python .agents/skills/compressor-experiment/scripts/run_and_log.py "
            f"--text \"{text}\" "
            f"--n_prompt {n_prompt} "
            f"--n_steps 50 "
            f"--exp_name {exp_name} "
            f"--run_id {run_id}"
        )
        run_cmd(cmd)
