from sys import argv
from settings import device
from diffusers import UnCLIPPipeline
import torch
import os

prompt = " ".join(argv[1:])

print("Prompt:", prompt)
print("Inference on", device.upper())

generate = UnCLIPPipeline.from_pretrained(
    "kakaobrain/karlo-v1-alpha",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

image = generate(prompt).images[0]

prompt = prompt.replace(" ", "-")
i = 0
while os.path.exists(f"outputs/{prompt}_{i}.png"):
    i += 1
filename = f"outputs/{prompt}_{i}.png"
image.save(filename)

print("Saved", filename)
print("Done!")
