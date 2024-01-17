import torch
import clip
import os
import matplotlib.pyplot as plt
import seaborn as sns

# CLIP using just first frame
# MAE student
# MAE decoded
# CLIP student
# CLIP decoded

# for each, try maximum and average


templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'a example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'a example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'a example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'a example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'a example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'a example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'a example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'a example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'a example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'a example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'a example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'a example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]

class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"
action_names = ['brushing hair', 'doing a cartwheel', 'catching', 'chewing', 'clapping', 'climbing', 'climbing stairs', 'diving', 'drawing a sword', 'dribbling', 'drinking', 'eating', 'falling to the floor', 'fencing', 'doing flic flac', 'golfing', 'doing a handstand', 'hitting', 'hugging', 'jumping', 'kicking', 'kicking a ball', 'kissing', 'laughing', 'picking', 'pouring', 'doing pullups', 'punching', 'pushing', 'doing pushups', 'riding a bike', 'riding a horse', 'running', 'shaking hands', 'shooting a ball', 'shooting a bow', 'shooting a gun', 'sitting', 'doing situps', 'smiling', 'smoking', 'doing a somersault', 'standing', 'swinging a baseball bat', 'using a sword', 'doing sword exercises', 'talking', 'throwing', 'turning', 'walking', 'waving']

# Convert the string to a list by splitting on spaces and then removing underscores
class_names = class_names_str.split()
# Sort the list alphabetically
class_names.sort()
print(class_names)


# each group of 48 correspond to one class, get index by integer division
prompts = {}
for name in action_names:
    if name not in prompts:
        prompts[name] = []
    for template in templates:
        prompts[name].append(template.format(name))


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

text_encodings = {}
model.eval()
with torch.no_grad():
    model.eval()

    count = 0
    for name in action_names:
        if name not in text_encodings:
            text_encodings[name] = []
        for prompt in prompts[name]:
            print(count)
            count += 1
            tokenized = clip.tokenize(prompt).to(device)
            print(tokenized.dtype)
            text_encoding = model.encode_text(tokenized)
            text_encodings[name].append(text_encoding)
        text_encodings[name] = torch.cat(text_encodings[name], dim=0)

# #normalize to unit vectors
# text_encodings /= text_encodings.norm(dim=-1, keepdim=True)

save_path = "action_encodings.pth"
torch.save(text_encodings, save_path)
print(f"Text encodings saved to {save_path}")

action_encodings = torch.cat(list(text_encodings.values()))

text_encodings = torch.load("text_encodings.pth")




cosine_similarities = torch.nn.functional.cosine_similarity(action_encodings.unsqueeze(0), text_encodings.unsqueeze(1), dim=-1)

print(cosine_similarities.shape)

# Set the figure size and create the heatmap
fig, ax = plt.subplots(figsize=(2448/100, 2448/100))
sns.heatmap(cosine_similarities.cpu().numpy(), cmap="viridis", xticklabels=False, yticklabels=False, cbar=True, ax=ax)

# Set the DPI to control the image size
dpi = 100
fig.set_dpi(dpi)

# Save the heatmap image with the desired resolution
heatmap_path = "cosine_similarity_heatmap.png"
plt.savefig(heatmap_path, dpi=dpi)
plt.close()

print(f"Heatmap saved to {heatmap_path}")








