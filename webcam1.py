import cv2
from ultralytics import YOLO
import threading
import json
import os
from collections import defaultdict
from threading import Thread

def run_face_and_object_detection(camera_id, vehicle_model_path, output_json_path):
    # Khởi tạo model nhận diện phương tiện
    vehicle_model = YOLO(vehicle_model_path)

    # Mở webcam theo ID camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Không thể mở webcam với ID {camera_id}!")
        return

    # Lấy thông số webcam
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Dictionary lưu thông tin tracking
    objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
    frame_count = 0

    print(f"Camera {camera_id} bắt đầu xử lý...")
    print(f"Độ phân giải: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(f"Không thể đọc frame từ webcam {camera_id}!")
            break

        # Phát hiện phương tiện
        vehicle_results = vehicle_model.track(frame, persist=True)

        # Xử lý kết quả vehicle detection
        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.cpu().numpy().astype(int)
            classes = vehicle_results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = vehicle_results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                if confidence < 0.85:  # Ngưỡng cho vehicle detection
                    continue

                class_name = vehicle_model.names[cls]
                if class_name not in ['car', 'truck', 'bus', 'motorcycle',"person"]:
                    continue

                if track_id not in objects[class_name]['tracks']:
                    objects[class_name]['count'] += 1
                    count = objects[class_name]['count']
                    objects[class_name]['tracks'][track_id] = {
                        'Object ID': f"{class_name.capitalize()} {count}",
                        'Class': class_name,
                        'Time_of_appearance': frame_count / fps,
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                    }
                else:
                    objects[class_name]['tracks'][track_id].update({
                        'Time_of_disappearance': frame_count / fps,
                        'bounding_box': box.tolist(),
                        'Confidence': float(confidence),
                    })

                # Vẽ vehicle detection
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name.capitalize()} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị video trong thời gian thực
        cv2.imshow(f'Webcam {camera_id} Real-time', frame)

        frame_count += 1

        # Thoát khi nhấn 'q' (mỗi camera có thể đóng riêng biệt bằng 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

    # Lưu kết quả vào JSON (nếu cần)
    json_results = []
    for class_name, data in objects.items():
        for track_id, info in data['tracks'].items():
            bounding_box = info.get('bounding_box', [0, 0, 0, 0])
            time_appeared = f"{int(info['Time_of_appearance'] // 60):02d}:{int(info['Time_of_appearance'] % 60):02d}"
            time_disappeared = f"{int(info['Time_of_disappearance'] // 60):02d}:{int(info['Time_of_disappearance'] % 60):02d}"

            json_results.append({
                "Object ID": info.get('Object ID', f"Vehicle {track_id}"),
                "Class": class_name,
                "Time appeared": time_appeared,
                "Time disappeared": time_disappeared,
                "Bounding box": bounding_box,
            })

    # Lưu kết quả JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)

    print(f"Đã lưu kết quả tracking của camera {camera_id} tại: {output_json_path}")

if __name__ == "__main__":
    camera_ids = [0]  # Thêm nhiều ID nếu có nhiều camera
    output_json_path = './output/webcam_output.json'  # Đường dẫn lưu file JSON kết quả
    vehicle_model_path = 'models/yolov10n.pt'  # Đường dẫn đến model vehicle detection

    threads = []

    # Tạo luồng cho mỗi camera
    for camera_id in camera_ids:
        thread = Thread(target=run_face_and_object_detection, args=(camera_id, vehicle_model_path, output_json_path))
        threads.append(thread)
        thread.start()

    # Chờ tất cả các luồng hoàn thành
    for thread in threads:
        thread.join()
    print("Tất cả các camera đã được xử lý xong.")
