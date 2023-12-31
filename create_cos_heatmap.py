import torch
import clip
import os
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "vid_encodings.pth"
#file_path = "img_encodings.pth"
#file_path = "text_encodings.pth"

video_encodings = torch.load(file_path)
video_encodings = video_encodings[:video_encodings.shape[0], ...]

vid_cosine_similarities = torch.nn.functional.cosine_similarity(video_encodings.unsqueeze(0),
                                                                video_encodings.unsqueeze(1), dim=-1)
print(vid_cosine_similarities.shape)
# Set the figure size and create the heatmap
fig, ax = plt.subplots(figsize=(3570 / 100, 3570 / 100))
sns.heatmap(vid_cosine_similarities.cpu().numpy(), cmap="viridis", xticklabels=False, yticklabels=False, cbar=True,
            ax=ax)

# Set the DPI to control the image size
dpi = 100
fig.set_dpi(dpi)
# Save the heatmap image with the desired resolution
heatmap_path = "video_cosine_similarity_heatmap.png"
plt.savefig(heatmap_path, dpi=dpi)
plt.close()
print(f"Heatmap saved to {heatmap_path}")