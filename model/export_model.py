import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

EXPORT_NAME = "dinov2_vitb14_feature.onnx"


class DinoFeature(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# default model: resnet50
def get_model(model_name):
    if model_name == "dinov2_vitb14":
        dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        model = DinoFeature(dinov2)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Identity()

    return model


model = get_model("dinov2_vitb14")
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy,
    EXPORT_NAME,
    input_names=["input"],
    output_names=["feature"],
    opset_version=17,
)
