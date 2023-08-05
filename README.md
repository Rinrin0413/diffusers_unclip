# Diffusers (UnCLIP)

## Setup [my environment (linux)]

```bash
# Create outputs directory
mkdir outputs

# Create venv
python3.10 -m venv .venv

# Activate venv
source .venv/bin/activate

# Install torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install diffusers, transformers
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
