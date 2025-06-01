import cv2
from ultralytics import YOLO
import cvzone
from PIL import Image
import numpy as np


class PeopleCounter:
    def __init__(self, model="detect/exp/weights/best.pt", tracker="trackers/botsort.yaml", device="cuda:0"):
        self.model = YOLO(model)
        self.tracker = tracker
        self.device = device

    def process_video(self, y: int, input_path: str, output_path: str = "output.mp4",
                      show: bool = True, save: bool = False, grey: bool = False) -> tuple[int, int]:
        """
        :param input_folder: Путь к папке с входными видеофайлами.
        :param output_folder: Путь к папке для выходных видео.
        :param y: Координата y первой линии.
        :param show: Показывать ли видео в реальном времени.
        :param save: Сохранять ли обработанные видео.
        :param grey: Преобразовывать ли видео в градации серого.
        """
        up = set()
        down = set()
        enter = set()
        exit = set()
        output_frames = []

        cap = cv2.VideoCapture(input_path)
        if show:
            cv2.namedWindow('Counter')
            cv2.setMouseCallback('Counter', self._mouse_callback)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if grey:
                pil_image = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame = cv2.cvtColor(
                    np.array(pil_image.convert("L")), cv2.COLOR_GRAY2BGR)

            results = self.model.track(
                frame, persist=True, classes=1, tracker=self.tracker, device=self.device, imgsz=640)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                    c = self.model.names[class_id]
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) // 2)
                    cy = int((y1 + y2) // 2)

                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

                    if cy < y and track_id not in down:
                        up.add(track_id)
                    if track_id in up and cy > y and track_id not in enter:
                        enter.add(track_id)

                    if cy > y and track_id not in up:
                        down.add(track_id)
                    if track_id in down and cy < y and track_id not in exit:
                        exit.add(track_id)

            cv2.line(frame, (1, y), (1024, y), (255, 255, 0), 2)
            cvzone.putTextRect(frame, f'enter:{len(enter)}', (50, 60), 2, 2)
            cvzone.putTextRect(frame, f'exit:{len(exit)}', (50, 100), 2, 2)

            if show:
                cv2.imshow("Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if save:
                output_frames.append(frame)

        cap.release()
        cv2.destroyAllWindows()
        if save:
            height, width, _ = output_frames[0].shape
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
                *'mp4v'), 30, (width, height))
            for frame in output_frames:
                out.write(frame)
            out.release()

        return len(enter), len(exit)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            point = [x, y]
            print(point)
