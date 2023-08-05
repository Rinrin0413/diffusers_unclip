from sys import argv
from settings import infer_on
from diffusers import UnCLIPPipeline
#from diffusers import UnCLIPImageVariationPipeline
import torch
import os
#import PIL

prompt = " ".join(argv[1:])

print("Prompt:", prompt)
print("Inference on", infer_on.upper())

torch_dtype = torch.float16 if infer_on == "cuda" else torch.float32

generate = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch_dtype).to(infer_on)
#variation = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch_dtype).to(infer_on)

image = generate(prompt).images[0]
#image_variabled = variation(image).images[0]

i = 0
while os.path.exists(f"outputs/{prompt}_{i}.png"):
    i += 1
image.save(f"outputs/{prompt}_{i}.png")

print("Done!")
