"""
UNETR model factory for BraTS 2021 segmentation (EXP-03, EXP-05, ABL-04 to ABL-06).

Architecture follows the original paper (Hatamizadeh et al., WACV 2022):
  - ViT-B/16 encoder: L=12 layers, K=768, 12 attention heads, MLP dim=3072
  - Patch resolution P=16 (16×16×16 voxels)
  - CNN decoder with skip connections from transformer layers {3, 6, 9, 12}
  - No pretrained weights (paper §4.3: "did not demonstrate any performance improvements")

MONAI 1.5 API note: 'pos_embed' was renamed to 'proj_type' in MONAI ≥1.3.
"""

from monai.networks.nets import UNETR


def get_unetr_model(
    img_size: tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 4,
    out_channels: int = 4,
    dropout_rate: float = 0.0,
) -> UNETR:
    """
    Returns the UNETR model as defined in the paper.

    Args:
        img_size:      Input patch size (same as training patch size). Default 128³.
        in_channels:   Number of input modalities. Default 4 (FLAIR, T1, T1ce, T2).
        out_channels:  Number of output classes. Default 4 (bg, NCR, ED, ET).
        dropout_rate:  Dropout probability. Paper uses 0.0.

    Output shape: (B, out_channels, H, W, D) — logits before softmax.
    """
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=16,          # CNN decoder feature size
        hidden_size=768,          # ViT-B hidden dimension
        mlp_dim=3072,             # ViT-B MLP dimension (= 4 × hidden_size)
        num_heads=12,             # ViT-B attention heads
        proj_type="conv",         # positional embedding type (MONAI 1.3+ API)
        norm_name="instance",     # instance normalisation in decoder
        res_block=True,           # residual blocks in decoder
        dropout_rate=dropout_rate,
        spatial_dims=3,
        qkv_bias=False,           # paper default
    )
