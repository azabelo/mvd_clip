import torch
import clip
import os

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
# Convert the string to a list by splitting on spaces and then removing underscores
class_names = class_names_str.split()
# Sort the list alphabetically
class_names.sort()
print(class_names)


# each group of 48 correspond to one class, get index by integer division
prompts = []
for name in class_names:
    for template in templates:
        prompts.append(template.format(name))


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

text_encodings = []
model.eval()
with torch.no_grad():
    model.eval()

    count = 0
    for prompt_batch in [prompts[i:i + 1] for i in range(0, len(prompts), 1)]:
        print(count)
        count += 1
        text_batch = clip.tokenize(prompt_batch).to(device)
        text_encoding = model.encode_text(text_batch)
        print(text_encoding.shape)
        text_encodings.append(text_encoding)

    text_encodings = torch.cat(text_encodings)

    save_path = "text_encodings.pth"
    torch.save(text_encodings, save_path)
    print(f"Text encodings saved to {save_path}")

print(model.transformer)
print(model.visual)
print(model.visual.proj.shape)

# create random image to pass to model.visual
image = torch.randn(1, 3, 224, 224).cuda()
image_encoding = model.encode_image(image)
print(image_encoding.shape)











