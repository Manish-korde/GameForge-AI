import time
from datasets import load_dataset

t0 = time.time()
print("Calling load_dataset streaming=True...")
ds = load_dataset("carlosuperb/lpc-action-pixel-art-diffusion", split="train", streaming=True)
print(f"Iterator initialized in {time.time()-t0:.2f}s. Fetching 3 items...")
for i, item in enumerate(ds):
    img = item['image']
    print(f"Item {i} received at {time.time()-t0:.2f}s, size={img.size}")
    if i >= 2:
        break
print("Done!")

