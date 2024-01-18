import torch
import clip
import os
import matplotlib.pyplot as plt
import seaborn as sns

templates = [
    'a photo of a person {}.',
    'a video of a person {}.',
    'an example of a person {}.',
    'a demonstration of a person {}.',
    'a photo of the person {}.',
    'a video of the person {}.',
    'an example of the person {}.',
    'a demonstration of the person {}.',
    'a photo of a person using {}.',
    'a video of a person using {}.',
    'an example of a person using {}.',
    'a demonstration of a person using {}.',
    'a photo of the person using {}.',
    'a video of the person using {}.',
    'an example of the person using {}.',
    'a demonstration of the person using {}.',
    'a photo of a person doing {}.',
    'a video of a person doing {}.',
    'an example of a person doing {}.',
    'a demonstration of a person doing {}.',
    'a photo of the person doing {}.',
    'a video of the person doing {}.',
    'an example of the person doing {}.',
    'a demonstration of the person doing {}.',
    'a photo of a person during {}.',
    'a video of a person during {}.',
    'an example of a person during {}.',
    'a demonstration of a person during {}.',
    'a photo of the person during {}.',
    'a video of the person during {}.',
    'an example of the person during {}.',
    'a demonstration of the person during {}.',
    'a photo of a person performing {}.',
    'a video of a person performing {}.',
    'an example of a person performing {}.',
    'a demonstration of a person performing {}.',
    'a photo of the person performing {}.',
    'a video of the person performing {}.',
    'an example of the person performing {}.',
    'a demonstration of the person performing {}.',
    'a photo of a person practicing {}.',
    'a video of a person practicing {}.',
    'an example of a person practicing {}.',
    'a demonstration of a person practicing {}.',
    'a photo of the person practicing {}.',
    'a video of the person practicing {}.',
    'an example of the person practicing {}.',
    'a demonstration of the person practicing {}.',
]

action_names = ['brushing hair', 'doing a cartwheel', 'catching', 'chewing', 'clapping', 'climbing', 'climbing stairs', 'diving', 'drawing a sword', 'dribbling', 'drinking', 'eating', 'falling to the floor', 'fencing', 'doing flic flac', 'golfing', 'doing a handstand', 'hitting', 'hugging', 'jumping', 'kicking', 'kicking a ball', 'kissing', 'laughing', 'picking', 'pouring', 'doing pullups', 'punching', 'pushing', 'doing pushups', 'riding a bike', 'riding a horse', 'running', 'shaking hands', 'shooting a ball', 'shooting a bow', 'shooting a gun', 'sitting', 'doing situps', 'smiling', 'smoking', 'doing a somersault', 'standing', 'swinging a baseball bat', 'using a sword', 'doing sword exercises', 'talking', 'throwing', 'turning', 'walking', 'waving']

# class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"
# # Convert the string to a list by splitting on spaces and then removing underscores
# class_names = class_names_str.split()
# # Sort the list alphabetically
# class_names.sort()


######## precompute text embeddings ######

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
            text_encoding = model.encode_text(tokenized)
            text_encodings[name].append(text_encoding)
        text_encodings[name] = torch.cat(text_encodings[name], dim=0)


action_encodings = torch.cat(list(text_encodings.values()))







