import torch
import clip
from timm.models import create_model






# class Alignment_Model(nn.Module):
#     def __init__(self, backbone, align_matrix_only=True):
#         super(Alignment_Model, self).__init__()
#         if os.path.exists("alignment_matrix.pth"):
#             alignment_matrix = torch.load("alignment_matrix.pth")
#         else:
#             clip_model, _ = clip.load("ViT-B/16", device="cuda")
#             # need to transpose it to give it to a linear layer
#             alignment_matrix = clip_model.visual.proj.float()
#
#         self.align_matrix_only = align_matrix_only
#         self.backbone = backbone
#         # Initialize a linear layer
#         self.linear_layer = nn.Linear(768, 512)
#         # Set the weight of the linear layer to the CLIP matrix
#         with torch.no_grad():
#             self.linear_layer.weight.copy_(alignment_matrix.t())
#
#     def forward(self, x):
#         empty_mask = torch.zeros((x.shape[0], 1568), dtype=torch.bool)
#         empty_mask = empty_mask.to('cuda', non_blocking=True)
#         # assuming that backbone is model.module
#         if self.align_matrix_only:
#             self.backbone.eval()
#             with torch.no_grad():
#                 self.backbone.eval()
#                 x = self.backbone.forward_encoder(x, empty_mask)
#                 x = x[:, 0, :]
#         else:
#             x = self.backbone.forward_encoder(x, empty_mask)
#             x = x[:, 0, :]
#         x = self.linear_layer(x)
#         return x
#
#     def retrieve_alignment_matrix(self):
#         return self.linear_layer.weight.data.t()
#
#
# action_embeddings = torch.load("action_encodings.pth")
# class_names_str = "brush_hair clap draw_sword fall_floor handstand kick pick push run shoot_gun smoke sword turn cartwheel climb dribble fencing hit kick_ball pour pushup shake_hands sit somersault sword_exercise walk catch climb_stairs drink flic_flac hug kiss pullup ride_bike shoot_ball situp stand talk wave chew dive eat golf jump laugh punch ride_horse shoot_bow smile swing_baseball throw"
# all_action_names = ['brushing hair', 'doing a cartwheel', 'catching', 'chewing', 'clapping', 'climbing',
#                 'climbing stairs', 'diving', 'drawing a sword', 'dribbling', 'drinking', 'eating',
#                 'falling to the floor', 'fencing', 'doing flic flac', 'golfing', 'doing a handstand',
#                 'hitting',
#                 'hugging', 'jumping', 'kicking', 'kicking a ball', 'kissing', 'laughing', 'picking',
#                 'pouring',
#                 'doing pullups', 'punching', 'pushing', 'doing pushups', 'riding a bike',
#                 'riding a horse',
#                 'running', 'shaking hands', 'shooting a ball', 'shooting a bow', 'shooting a gun',
#                 'sitting',
#                 'doing situps', 'smiling', 'smoking', 'doing a somersault', 'standing',
#                 'swinging a baseball bat',
#                 'using a sword', 'doing sword exercises', 'talking', 'throwing', 'turning', 'walking',
#                 'waving']
# all_class_names = class_names_str.split()

