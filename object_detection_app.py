### this code is using fast cnn pre-trained model
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights
)
from torchvision import transforms

# 1) Load a pretrained detector once
@st.cache_resource
def load_detector():
    weights  = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model    = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.3)
    model.eval()
    return model, weights.meta["categories"]

detector, categories = load_detector()

# 2) Title + uploader
st.title("🚲 Bicycle-Only Detector")
upload = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if not upload:
    st.info("Please upload an image.")
    st.stop()

# 3) Open + preprocess
img = Image.open(upload).convert("RGB")
transform = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
img_t = transform(img).unsqueeze(0)  # (1,3,H,W)

# 4) Run detection
with torch.no_grad():
    preds = detector(img_t)[0]

# 5) Filter for bicycles
bike_indices = [
    i for i, lab in enumerate(preds["labels"])
    if categories[lab] == "bicycle"
]

# 6) Draw them
img_out = img.copy()
draw   = ImageDraw.Draw(img_out)
font   = ImageFont.load_default()

if not bike_indices:
    st.warning("No bicycle detected with confidence ≥ threshold.")
else:
    for i in bike_indices:
        x0, y0, x1, y1 = preds["boxes"][i].round().tolist()
        score = preds["scores"][i].item()
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0 - 10),
                  f"bike {score:.2f}",
                  font=font,
                  fill="red")

# 7) Display
st.image(img_out, use_container_width=True)



