# Diffusers (UnCLIP)

## Setup [my environment (linux)]

```bash
git clone https://github.com/Rinrin0413/diffusers_unclip.git
cd diffusers_unclip
mkdir outputs

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install diffusers[torch] transformers
```

## Usage (linux)

```bash
# If not already activated
source .venv/bin/activate
```

### Text to Image (t2i)

```bash
# Run script with prompts
python . carbonara on a black plate
```

### Image Variation (i2i)

```bash
# Run script with image path
python variation.py ./outputs/carbonara-on-a-black-plate_0.png
```
