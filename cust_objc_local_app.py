import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from model.model import BBoxClassifierCNN

@st.cache_resource
def load_model():
    model = BBoxClassifierCNN(img_size=256, num_classes=2)
    ckpt  = 'checkpoints/bbox_cnn_cls.pth'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])

st.title("🚲 Bike Detector + Localizer")
upload = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
if not upload:
    st.stop()

img = Image.open(upload).convert("RGB")
ow, oh = img.size
inp = transform(img).unsqueeze(0)

with torch.no_grad():
    bbox_out, cls_logits = model(inp)
    probs = F.softmax(cls_logits, dim=1)[0]
    conf  = probs[1].item()   # class 1 = bicycle

if conf > 0.5:
    # denormalize center/wh → corners
    cx, cy, w, h = bbox_out.squeeze().cpu().numpy()
    xmin = int((cx - w/2)*ow); xmax = int((cx + w/2)*ow)
    ymin = int((cy - h/2)*oh); ymax = int((cy + h/2)*oh)
    xmin, xmax = sorted((xmin,xmax)); ymin, ymax = sorted((ymin,ymax))
    xmin, ymin = max(0,xmin), max(0,ymin)
    xmax, ymax = min(ow,xmax), min(oh,ymax)

    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    font = ImageFont.load_default()
    draw.rectangle([xmin,ymin,xmax,ymax], outline="red", width=3)
    draw.text((xmin, ymin-12), f"bike {conf:.2f}", font=font, fill="red")

    st.image(img2, use_container_width=True)
else:
    st.warning("No bicycle detected (conf < 0.5).")
