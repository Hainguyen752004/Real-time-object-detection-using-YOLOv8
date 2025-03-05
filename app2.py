import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

def process_video(video_path, model_path, display_placeholder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Không thể mở video!")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Tạo file video tạm để lưu kết quả
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Load YOLO model
    model = YOLO(model_path)

    frame_count = 0
    progress_bar = st.progress(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.track(frame, persist=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                if confidence < 0.85:
                    continue
                
                class_name = model.names[cls]
                if class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name.capitalize()} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)

        # Cập nhật khung hình lên Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        frame_count += 1
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    out.release()
    return output_path

st.title("🚗 Phát hiện phương tiện theo thời gian thực với YOLO")

uploaded_video = st.file_uploader("📂 Chọn video để xử lý", type=["mp4", "avi", "mov"])
model_path = "models/yolov10n.pt"

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        video_path = temp_video.name
        
    st.video(video_path)

    if st.button("▶ Bắt đầu xử lý video"):
        st.write("⏳ Đang xử lý video, vui lòng chờ...")
        display_placeholder = st.empty()  # Khu vực hiển thị frame theo thời gian thực
        output_video_path = process_video(video_path, model_path, display_placeholder)

        if output_video_path:
            st.success("✅ Xử lý hoàn tất!")
            st.video(output_video_path)
