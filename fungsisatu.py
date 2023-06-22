import streamlit as st
import torch
import os
import cv2
from PIL import Image
import numpy as np
import pytesseract
from utils.general import non_max_suppression
import re

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.dataloaders import LoadImages

@st.cache(allow_output_mutation=True)
@st.cache(allow_output_mutation=True)
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'yolov5', 'runs', 'train', 'exp9', 'weights', 'best.pt')
    model = DetectMultiBackend(model_path, device=select_device(''), dnn=False, data='data/coco128.yaml', fp16=False)
    return model


@st.cache(allow_output_mutation=True)
def read_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def bacaannya_apa(ini_, model):
    ini = cv2.resize(ini_, (640, 640))
    test = []
    test = np.array([ini[:,:,0], ini[:,:,1], ini[:,:,2]])
    ini = torch.from_numpy(test).to(model.device)
    ini = ini.half() if model.fp16 else ini.float()
    ini /= 255
    if len(ini.shape) == 3:
        ini = ini[None]  # expand for batch dim

    pred = model(ini, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, max_det=1000)

    tinggi = ini_.shape[0] / 640
    lebar = ini_.shape[1] / 640

    hasil = []
    for i, prediksi in enumerate(pred[0]):
        y1, x1, y2, x2 = prediksi[0] * lebar, prediksi[1] * tinggi, prediksi[2] * lebar, prediksi[3] * tinggi
        beda = ini_[int(x1) - 25:int(x2) + 10, int(y1) - 40:int(y2) + 5] * 255
        try:
            new_p = Image.fromarray(beda)
        except:
            continue
        bacaannya = pytesseract.image_to_string(new_p)
        bacaannya = re.sub('[^A-Z0-9 ]', '', bacaannya)
        kelas_prediksi = model.names[int(prediksi[-1])]
        hasil.append([kelas_prediksi, bacaannya])
        st.image(new_p, caption=bacaannya)
    
    return hasil

# Main Streamlit app
def main():
    st.title("Deteksi e-KTP")
    
    # Load YOLOv5 model
    model = load_model()
    
    st.write('halo')

if __name__ == "__main__":
    main()
