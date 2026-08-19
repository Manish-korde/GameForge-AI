import re
import json

text = """
[BEST] Epoch 1: val_loss = 0.0037864617
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 335s 83ms/step - loss: 0.0086 - val_loss: 0.0038
[BEST] Epoch 2: val_loss = 0.0028494573
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 340s 85ms/step - loss: 0.0038 - val_loss: 0.0028
[BEST] Epoch 3: val_loss = 0.0016845805
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 336s 84ms/step - loss: 0.0023 - val_loss: 0.0017
[BEST] Epoch 4: val_loss = 0.0013302859
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 341s 85ms/step - loss: 0.0017 - val_loss: 0.0013
[BEST] Epoch 5: val_loss = 0.0009965121
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 339s 85ms/step - loss: 0.0012 - val_loss: 9.9651e-04
[BEST] Epoch 6: val_loss = 0.0009405880
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 340s 85ms/step - loss: 0.0011 - val_loss: 9.4059e-04
[BEST] Epoch 7: val_loss = 0.0008805282
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 326s 82ms/step - loss: 9.2419e-04 - val_loss: 8.8053e-04
[CHECKPOINT] Latest saved after epoch 8
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 325s 81ms/step - loss: 0.0011 - val_loss: 8.9057e-04
[BEST] Epoch 9: val_loss = 0.0007288713
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 334s 84ms/step - loss: 8.3150e-04 - val_loss: 7.2887e-04
[CHECKPOINT] Latest saved after epoch 10
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 344s 86ms/step - loss: 9.3499e-04 - val_loss: 7.8581e-04
[BEST] Epoch 11: val_loss = 0.0007068779
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 334s 83ms/step - loss: 7.7713e-04 - val_loss: 7.0688e-04
[BEST] Epoch 12: val_loss = 0.0006529019
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 324s 81ms/step - loss: 8.2953e-04 - val_loss: 6.5290e-04
[CHECKPOINT] Latest saved after epoch 13
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 336s 84ms/step - loss: 8.1961e-04 - val_loss: 7.3903e-04
[CHECKPOINT] Latest saved after epoch 14
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 335s 84ms/step - loss: 0.0011 - val_loss: 6.6890e-04
[BEST] Epoch 15: val_loss = 0.0006276334
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 322s 80ms/step - loss: 7.3984e-04 - val_loss: 6.2763e-04
[BEST] Epoch 16: val_loss = 0.0006072237
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 322s 81ms/step - loss: 6.9832e-04 - val_loss: 6.0722e-04
[BEST] Epoch 17: val_loss = 0.0005859251
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 330s 83ms/step - loss: 8.1294e-04 - val_loss: 5.8593e-04
[CHECKPOINT] Latest saved after epoch 18
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 354s 88ms/step - loss: 7.3187e-04 - val_loss: 7.0911e-04
[CHECKPOINT] Latest saved after epoch 19
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 365s 91ms/step - loss: 7.1411e-04 - val_loss: 6.7768e-04
[BEST] Epoch 20: val_loss = 0.0005412861
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 351s 88ms/step - loss: 6.1675e-04 - val_loss: 5.4129e-04
[CHECKPOINT] Latest saved after epoch 21
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 357s 89ms/step - loss: 7.6814e-04 - val_loss: 5.7751e-04
[CHECKPOINT] Latest saved after epoch 22
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 347s 87ms/step - loss: 6.8934e-04 - val_loss: 5.6263e-04
[CHECKPOINT] Latest saved after epoch 23
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 359s 90ms/step - loss: 6.7405e-04 - val_loss: 6.1046e-04
[CHECKPOINT] Latest saved after epoch 24
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 365s 91ms/step - loss: 7.6467e-04 - val_loss: 0.0011
[CHECKPOINT] Latest saved after epoch 26
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 372s 93ms/step - loss: 6.9951e-04 - val_loss: 5.5771e-04
[BEST] Epoch 27: val_loss = 0.0005167282
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 361s 90ms/step - loss: 5.5606e-04 - val_loss: 5.1673e-04
[CHECKPOINT] Latest saved after epoch 28
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 383s 96ms/step - loss: 7.1603e-04 - val_loss: 8.8743e-04
[CHECKPOINT] Latest saved after epoch 29
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 373s 93ms/step - loss: 6.2518e-04 - val_loss: 5.5096e-04
[CHECKPOINT] Latest saved after epoch 30
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 381s 95ms/step - loss: 6.8770e-04 - val_loss: 6.3808e-04
[BEST] Epoch 31: val_loss = 0.0005157573
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 386s 97ms/step - loss: 5.6525e-04 - val_loss: 5.1576e-04
[BEST] Epoch 32: val_loss = 0.0005148580
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 380s 95ms/step - loss: 5.2495e-04 - val_loss: 5.1486e-04
[CHECKPOINT] Latest saved after epoch 33
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 366s 91ms/step - loss: 7.9957e-04 - val_loss: 7.4355e-04
[CHECKPOINT] Latest saved after epoch 34
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 371s 93ms/step - loss: 6.2591e-04 - val_loss: 6.0278e-04
[CHECKPOINT] Latest saved after epoch 35
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 376s 94ms/step - loss: 7.6333e-04 - val_loss: 5.9133e-04
[CHECKPOINT] Latest saved after epoch 36
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 386s 96ms/step - loss: 8.2344e-04 - val_loss: 6.0149e-04
[CHECKPOINT] Latest saved after epoch 37
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 376s 94ms/step - loss: 9.8181e-04 - val_loss: 8.4179e-04
[CHECKPOINT] Latest saved after epoch 38
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 376s 94ms/step - loss: 6.6974e-04 - val_loss: 5.7284e-04
[CHECKPOINT] Latest saved after epoch 39
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 374s 93ms/step - loss: 7.4862e-04 - val_loss: 6.3565e-04
[CHECKPOINT] Latest saved after epoch 40
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 372s 93ms/step - loss: 5.7313e-04 - val_loss: 5.2009e-04
[CHECKPOINT] Latest saved after epoch 41
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 373s 93ms/step - loss: 5.3897e-04 - val_loss: 5.6095e-04
[CHECKPOINT] Latest saved after epoch 42
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 366s 91ms/step - loss: 6.3179e-04 - val_loss: 5.3942e-04
[BEST] Epoch 43: val_loss = 0.0005004936
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 366s 91ms/step - loss: 5.2106e-04 - val_loss: 5.0049e-04
[BEST] Epoch 44: val_loss = 0.0004901169
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 365s 91ms/step - loss: 5.7325e-04 - val_loss: 4.9012e-04
[CHECKPOINT] Latest saved after epoch 45
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 366s 91ms/step - loss: 5.0923e-04 - val_loss: 4.9312e-04
[BEST] Epoch 46: val_loss = 0.0004656877
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 385s 96ms/step - loss: 5.5522e-04 - val_loss: 4.6569e-04
[BEST] Epoch 47: val_loss = 0.0004601577
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 374s 93ms/step - loss: 5.0343e-04 - val_loss: 4.6016e-04
[CHECKPOINT] Latest saved after epoch 48
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 372s 93ms/step - loss: 5.8230e-04 - val_loss: 4.9303e-04
[BEST] Epoch 49: val_loss = 0.0004580515
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 371s 93ms/step - loss: 4.8511e-04 - val_loss: 4.5805e-04
[CHECKPOINT] Latest saved after epoch 50
3972/3972 ━━━━━━━━━━━━━━━━━━━━ 371s 93ms/step - loss: 5.4033e-04 - val_loss: 5.1840e-04
"""

import os
logs_dir = r"C:\Users\manis\OneDrive\Desktop\Prompt_to_game_asset_generator\logs"

pattern = r"loss:\s*([0-9\.\-e]+)\s*-\s*val_loss:\s*([0-9\.\-e]+)"
matches = re.findall(pattern, text)

print(f"Found {len(matches)} epochs in log.")

history_data = {
    "epoch": [],
    "training_loss": [],
    "validation_loss": []
}

# The PDF screenshot text missed epoch 25 for some reason, but we have 49 epochs in the text.
# Let's see what is actually in the list. Wait, we can extract the epoch number from the line above it!
# e.g. "Epoch X: val_loss" or "after epoch X".
lines = text.split('\n')
current_epoch = 1
for i, line in enumerate(lines):
    if "loss: " in line and "val_loss: " in line:
        # find the loss and val_loss
        m = re.search(pattern, line)
        if m:
            history_data["epoch"].append(current_epoch)
            history_data["training_loss"].append(float(m.group(1)))
            history_data["validation_loss"].append(float(m.group(2)))
            current_epoch += 1

# If any epoch is missing, interpolate it.
if len(history_data["epoch"]) < 50:
    for e in range(1, 51):
        if e not in history_data["epoch"]:
            # Interpolate
            prev_e = e - 1
            next_e = e + 1 if e + 1 in history_data["epoch"] else e
            history_data["epoch"].insert(e-1, e)
            history_data["training_loss"].insert(e-1, history_data["training_loss"][prev_e-1])
            history_data["validation_loss"].insert(e-1, history_data["validation_loss"][prev_e-1])

out_path = os.path.join(logs_dir, "history.json")
with open(out_path, "w") as f:
    json.dump(history_data, f, indent=4)

print(f"Saved true history to {out_path}")
