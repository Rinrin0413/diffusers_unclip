from sys import argv
from settings import device
from diffusers import UnCLIPImageVariationPipeline
import torch
from PIL import Image
import os

path = argv[1]

print("Inference on", device.upper())
print("Generating variation of", path)

variation = UnCLIPImageVariationPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha-image-variations",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

image = Image.open(path)
image = variation(image).images[0]

path = path.replace(".png", "")
i = 0
while os.path.exists(f"{path}_variation{i}.png"):
    i += 1
filename = f"{path}_variation{i}.png"
image.save(filename)

print("Saved", filename)
print("Done!")
