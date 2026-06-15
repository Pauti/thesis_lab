"""
Multi-task UNETR for RQ3 (EXP-05): segmentation + auxiliary ET-volume ordinal
classification.

The auxiliary head taps the ViT bottleneck (the final transformer output z12,
shape (B, N=512, hidden=768)), applies global average pooling over the N tokens,
and predicts the ET-volume tertile via a CORAL ordinal head:

    GAP(z12) -> Linear(768, 256) -> ReLU -> CoralLayer(256, K)

Tapping the encoder bottleneck (rather than a decoder feature) is what makes the
auxiliary signal regularise the *encoder*; it is also the direct analogue of the
nnU-Net encoder bottleneck used for the later nnU-Net multi-task variant, which
keeps the CNN-vs-Transformer comparison fair.

forward(x, return_aux=False) returns segmentation logits only by default so that
MONAI's sliding_window_inference (which expects a single tensor) works unchanged
during validation/inference. Pass return_aux=True in the training loop to also
get the ordinal logits.
"""

import torch
import torch.nn as nn
from monai.networks.nets import UNETR

from utils.ordinal import CoralLayer


class MultiTaskUNETR(UNETR):
    """UNETR with an auxiliary CORAL ordinal-classification head on the ViT bottleneck."""

    def __init__(
        self,
        *,
        num_ordinal_classes: int = 3,
        aux_hidden: int = 256,
        **unetr_kwargs,
    ):
        super().__init__(**unetr_kwargs)
        hidden_size = unetr_kwargs.get("hidden_size", 768)
        self.num_ordinal_classes = num_ordinal_classes
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, aux_hidden),
            nn.ReLU(inplace=True),
            CoralLayer(aux_hidden, num_ordinal_classes),
        )

    def forward(self, x_in: torch.Tensor, return_aux: bool = False):
        # Body copied from monai.networks.nets.UNETR.forward (MONAI 1.5.2),
        # extended to expose the bottleneck `x` for the auxiliary head.
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        seg_logits = self.out(out)

        if not return_aux:
            return seg_logits

        pooled = x.mean(dim=1)            # GAP over N tokens -> (B, hidden_size)
        aux_logits = self.aux_head(pooled)  # (B, K-1)
        return seg_logits, aux_logits


def get_multitask_unetr_model(
    img_size: tuple[int, int, int] = (128, 128, 128),
    in_channels: int = 4,
    out_channels: int = 4,
    dropout_rate: float = 0.0,
    num_ordinal_classes: int = 3,
    aux_hidden: int = 256,
) -> MultiTaskUNETR:
    """
    Returns the multi-task UNETR. Segmentation backbone is identical to
    get_unetr_model() (same ViT-B/16 config) so the only added parameters are
    the auxiliary head — keeping EXP-05 comparable to the EXP-03/ABL-07 baseline.
    """
    return MultiTaskUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=img_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=dropout_rate,
        spatial_dims=3,
        qkv_bias=False,
        num_ordinal_classes=num_ordinal_classes,
        aux_hidden=aux_hidden,
    )
