import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


def create_model(
    freeze_backbone: bool = True,
    unfreeze_last_block: bool = False,
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    if unfreeze_last_block:
        for parameter in model.features[-1].parameters():
            parameter.requires_grad = True

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features=input_features,
        out_features=1,
    )

    return model


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = create_model(freeze_backbone=True)
    model = model.to(device)

    dummy_batch = torch.randn(
        8,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        logits = model(dummy_batch)

    print(f"Device: {device}")
    print(f"Input shape: {dummy_batch.shape}")
    print(f"Output shape: {logits.shape}")
    print(model.classifier)