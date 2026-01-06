# StoryReasoning Multimodal Project — Autoencoder (Fixed Gradient)
Fixes “black predictions” by allowing gradients through AE.decode.

## Run
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
python -m src.pretrain_ae --config config.yaml --epochs 15
python -m src.train --config config.yaml
python -m src.eval --config config.yaml


## Troubleshooting
- If `ModuleNotFoundError: src`: run commands with `python -m ...` from the project root.
- If images are black: confirm you're using this fixed build and retrain.
- If HF download is slow: `pip install -U datasets` and try again.
