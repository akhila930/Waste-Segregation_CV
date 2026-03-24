import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
import time
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="♻️ AI Waste Segregation", layout="wide")

st.title("♻️ AI Waste Segregation Assistant")
st.caption("CNN vs ResNet Comparison Dashboard")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# CNN MODEL (MATCH TRAINING EXACTLY)
# -------------------------------
class WasteCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = torch.nn.Conv2d(3,32,3,padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        
        self.conv2 = torch.nn.Conv2d(32,64,3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        
        self.conv3 = torch.nn.Conv2d(64,128,3,padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        
        self.conv4 = torch.nn.Conv2d(128,256,3,padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        
        self.pool = torch.nn.MaxPool2d(2,2)
        
        self.fc1 = torch.nn.Linear(256*14*14,256)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(256,2)

    def forward(self,x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    
    # ResNet
    resnet = models.resnet18(pretrained=False)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load("../models/best.pt", map_location=device))
    resnet.to(device).eval()

    # CNN (FIXED)
    cnn = WasteCNN()
    cnn.load_state_dict(torch.load("../models/cnn_best.pt", map_location=device))
    cnn.to(device).eval()

    # YOLO
    yolo = YOLO("yolov8n.pt")

    return resnet, cnn, yolo

resnet, cnn, yolo = load_models()

# -------------------------------
# TRANSFORMS
# -------------------------------
resnet_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

cnn_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

classes = ["Biodegradable","Non-Biodegradable"]

# -------------------------------
# PREDICTION FUNCTION
# -------------------------------
def predict(model, img, tf):
    img = Image.fromarray(img).convert("RGB")
    img = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(prob,1)

    return classes[pred.item()], conf.item()

# -------------------------------
# SIDEBAR
# -------------------------------
mode = st.sidebar.radio("Select Mode", ["Upload Image","Webcam"])

# -------------------------------
# UPLOAD MODE
# -------------------------------
if mode == "Upload Image":
    file = st.file_uploader("Upload Image")

    if file:
        image = Image.open(file)
        img_np = np.array(image)

        st.image(image, caption="Input Image")

        results = yolo(img_np)

        rows = []

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            names = r.names
            cls_ids = r.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, cls_ids):
                x1,y1,x2,y2 = map(int, box)
                crop = img_np[y1:y2, x1:x2]

                obj = names[int(cls)]

                r_label, r_conf = predict(resnet, crop, resnet_tf)
                c_label, c_conf = predict(cnn, crop, cnn_tf)

                rows.append({
                    "Object": obj,
                    "ResNet": f"{r_label} ({r_conf*100:.1f}%)",
                    "CNN": f"{c_label} ({c_conf*100:.1f}%)"
                })

                cv2.rectangle(img_np,(x1,y1),(x2,y2),(0,255,0),2)

        st.image(img_np, caption="Detection")

        st.markdown("### 📊 Model Comparison")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# -------------------------------
# WEBCAM MODE (10 sec)
# -------------------------------
elif mode == "Webcam":

    if st.button("Start 10-sec Scan"):
        cap = cv2.VideoCapture(0)
        start = time.time()

        data = []
        FRAME = st.image([])

        while time.time() - start < 10:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame)

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                names = r.names
                cls_ids = r.boxes.cls.cpu().numpy()

                for box, cls in zip(boxes, cls_ids):
                    x1,y1,x2,y2 = map(int, box)
                    crop = frame[y1:y2, x1:x2]

                    obj = names[int(cls)]

                    r_label,_ = predict(resnet, crop, resnet_tf)
                    c_label,_ = predict(cnn, crop, cnn_tf)

                    data.append((obj, r_label, c_label))

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            FRAME.image(frame, channels="BGR")

        cap.release()

        st.success("Scan Complete ✅")

        if data:
            df = pd.DataFrame(data, columns=["Object","ResNet","CNN"])
            st.dataframe(df)