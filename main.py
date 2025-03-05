import os
import cv2
from ultralytics import YOLO
import json
from collections import defaultdict

class VideoProcessor:
    def __init__(self, video_path, vehicle_model_path, output_video_path, output_json_path):
        self.video_path = video_path
        self.vehicle_model_path = vehicle_model_path
        self.output_video_path = output_video_path
        self.output_json_path = output_json_path
        self.objects = defaultdict(lambda: {'count': 0, 'tracks': {}})
        self.frame_count = 0

    def initialize(self):
        """Khởi tạo các thành phần cần thiết."""
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'true'
        self._create_output_directory()
        self._check_input_file()
        self.vehicle_model = YOLO(self.vehicle_model_path)
        self.cap = self._open_video()
        self.frame_width, self.frame_height, self.fps, self.total_frames = self._get_video_properties()
        self.out = self._initialize_video_writer()

    def _create_output_directory(self):
        """Tạo thư mục đầu ra nếu chưa tồn tại."""
        output_dir = os.path.dirname(self.output_video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _check_input_file(self):
        """Kiểm tra xem file video đầu vào có tồn tại không."""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Không tìm thấy file video: {self.video_path}")

    def _open_video(self):
        """Mở video và kiểm tra xem có thể đọc được không."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video: {self.video_path}")
        return cap

    def _get_video_properties(self):
        """Lấy các thông số cơ bản của video."""
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_width, frame_height, fps, total_frames

    def _initialize_video_writer(self):
        """Khởi tạo video writer để lưu video đầu ra."""
        return cv2.VideoWriter(
            self.output_video_path,
            cv2.VideoWriter_fourcc(*'X264'),
            self.fps,
            (self.frame_width, self.frame_height)
        )

    def process_frame(self, frame):
        """Xử lý từng frame để phát hiện phương tiện."""
        vehicle_results = self.vehicle_model.track(frame, persist=True)

        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xyxy.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.cpu().numpy().astype(int)
            classes = vehicle_results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = vehicle_results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                if confidence < 0.85:  # Ngưỡng cho vehicle detection
                    continue

                class_name = self.vehicle_model.names[cls]
                if class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                    continue

                self._update_tracking_info(class_name, track_id, box, confidence)
                self._draw_vehicle_info(frame, box, class_name, confidence)

    def _update_tracking_info(self, class_name, track_id, box, confidence):
        """Cập nhật thông tin tracking cho phương tiện."""
        if track_id not in self.objects[class_name]['tracks']:
            self.objects[class_name]['count'] += 1
            count = self.objects[class_name]['count']
            self.objects[class_name]['tracks'][track_id] = {
                'Object ID': f"{class_name.capitalize()} {count}",
                'Class': class_name,
                'Time_of_appearance': self.frame_count / self.fps,
                'Time_of_disappearance': self.frame_count / self.fps,
                'bounding_box': box.tolist(),
                'Confidence': float(confidence),
            }
        else:
            self.objects[class_name]['tracks'][track_id].update({
                'Time_of_disappearance': self.frame_count / self.fps,
                'bounding_box': box.tolist(),
                'Confidence': float(confidence),
            })

    def _draw_vehicle_info(self, frame, box, class_name, confidence):
        """Vẽ thông tin phương tiện lên frame."""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name.capitalize()} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def save_results(self):
        """Lưu kết quả tracking vào file JSON."""
        json_results = []
        for class_name, data in self.objects.items():
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

        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=4)

    def run(self):
        """Chạy quá trình xử lý video."""
        self.initialize()
        print(f"Bắt đầu xử lý video: {os.path.basename(self.video_path)}")
        print(f"Độ phân giải: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps}")

        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            self.process_frame(frame)
            cv2.imshow('Processed Video', frame)
            self.out.write(frame)

            self.frame_count += 1
            if self.frame_count % 60 == 0:
                progress = (self.frame_count / self.total_frames) * 100
                print(f"\rXử lý: {progress:.2f}% hoàn thành", end="")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        self.save_results()
        print(f"\nĐã lưu video đã xử lý tại: {self.output_video_path}")
        print(f"Đã lưu kết quả tracking tại: {self.output_json_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xử lý video với vehicle detection")
    parser.add_argument('--input', type=str, required=True, help="Đường dẫn đến file video input")
    parser.add_argument('--output_dir', type=str, default='./output', help="Thư mục lưu kết quả")
    parser.add_argument('--vehicle_model', type=str, default='models/yolov10s.pt', help="Đường dẫn đến model phát hiện phương tiện")
    args = parser.parse_args()
    
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_video = os.path.join(args.output_dir, f"{base_name}_Out_vehicle.mp4")
    output_json = os.path.join(args.output_dir, f"{base_name}_Out_vehicle.json")
    
    processor = VideoProcessor(
        video_path=args.input,
        vehicle_model_path=args.vehicle_model,
        output_video_path=output_video,
        output_json_path=output_json
    )
    processor.run()