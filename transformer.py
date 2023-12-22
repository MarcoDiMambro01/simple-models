import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import cv2

#plot attention maps
def get_attention_map(img, model, get_mask=False):
    #x = transform(img)
    #x.size()
    x=img
    logits, att_mat = model(x.unsqueeze(0))

    att_mat = torch.stack(att_mat).squeeze(1)

    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
    if get_mask:
        result = cv2.resize(mask / mask.max(), img.size)
    else:        
        mask = cv2.resize(mask / mask.max(), img.size)[..., np.newaxis]
        result = (mask * img).astype("uint8")
    
    return result





class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        attn_output, attn_weights = self.attn(inp_x, inp_x, inp_x)
        x = x + attn_output
        x = x + self.linear(self.layer_norm_2(x))
        return x, attn_weights
    

def img_to_patch(x, patch_size, flatten_channels = True):
        """
        Inputs:
            x - torch.Tensor representing the image of shape [B, C, H, W]
            patch_size - Number of pixels per dimension of the patches (integer)
            flatten_channels - If True, the patches will be returned in a flattened format
                               as a feature vector instead of a image grid.
        """
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
        x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
        if flatten_channels:
            x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
        return x
    
