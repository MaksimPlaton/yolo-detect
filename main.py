import cv2
from ultralytics import YOLO
import cvzone


class PeopleCounter:
    def __init__(self, model_name="yolo11m.pt", tracker="bytetrack.yaml", device="cuda:0"):
        self.model = YOLO(f'yolo_models/{model_name}')
        self.names = self.model.model.names
        self.tracker = tracker
        self.device = device

    def process_video(self, cy1, cy2, video_path, output_path="output.mp4"):
        up = set()
        down = set()
        enter = set()
        exit = set()
        output_frames = []

        cap = cv2.VideoCapture(video_path)
        cv2.namedWindow('RGB')
        cv2.setMouseCallback('RGB', self._mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1020, 600))
            results = self.model.track(
                frame, persist=True, classes=0, tracker=self.tracker, device=self.device)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                    c = self.names[class_id]
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) // 2)
                    cy = int((y1 + y2) // 2)

                    # Логика подсчета людей
                    if cy < cy1 and track_id not in down:
                        up.add(track_id)
                    if track_id in up and cy > cy2 and track_id not in enter:
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(
                            frame, f'{track_id}', (x1, y2), 1, 1)
                        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                        enter.add(track_id)

                    if cy > cy2 and track_id not in up:
                        down.add(track_id)
                    if track_id in down and cy < cy1 and track_id not in exit:
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)
                        cvzone.putTextRect(
                            frame, f'{track_id}', (x1, y2), 1, 1)
                        cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                        exit.add(track_id)

            cv2.line(frame, (1, cy2), (1024, cy2), (0, 255, 0), 2)
            cv2.line(frame, (1, cy1), (1024, cy1), (255, 255, 0), 2)
            cvzone.putTextRect(frame, f'enter:{len(enter)}', (50, 60), 2, 2)
            cvzone.putTextRect(frame, f'exit:{len(exit)}', (50, 100), 2, 2)

            output_frames.append(frame)
            cv2.imshow("RGB", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        height, width, _ = output_frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
            *'mp4v'), 30, (width, height))
        for frame in output_frames:
            out.write(frame)
        out.release()

    def _mouse_callback(self, event, x, y, flags, param):
        """Обработчик событий мыши."""
        if event == cv2.EVENT_MOUSEMOVE:
            point = [x, y]
            print(point)


if __name__ == "__main__":
    counter = PeopleCounter()
    counter.process_video(380, 390,
                          video_path="input_videos/test_1.mp4", output_path="output_1.mp4")
