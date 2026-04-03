python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy ipykernel click Pillow psutil requests scipy tqdm wandb matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu124
