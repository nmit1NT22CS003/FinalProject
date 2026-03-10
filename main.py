import cv2
import numpy as np
import pyttsx3
import time
import psutil

from yolo_utils import infer_image
from customdata import REAL_WIDTHS


CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
FOCAL_LENGTH = 900
ENABLE_SPEECH = True


def load_yolo():

    labels = open("coco.names").read().strip().split("\n")

    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(labels), 3))

    return net, labels, colors, output_layers


def speak(text, engine):

    start = time.time()

    engine.say(text)
    engine.runAndWait()

    latency = time.time() - start
    print(f"[Speech latency]: {latency:.3f} sec")


def main():

    class FLAGS:
        confidence = CONF_THRESHOLD
        threshold = NMS_THRESHOLD
        show_time = False


    net, labels, colors, output_layers = load_yolo()

    focal_length = FOCAL_LENGTH

    print(f"[INFO] Using focal length: {focal_length:.2f}")


    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return


    engine = pyttsx3.init() if ENABLE_SPEECH else None
    last_speech = 0


    # Metrics
    total_conf = 0
    total_detections = 0


    print("[INFO] Starting webcam. Press 'q' to quit.")


    while True:

        start_time = time.time()

        ret, frame = cap.read()

        if not ret:
            break


        h, w = frame.shape[:2]

        frame, boxes, confs, classids, idxs = infer_image(
            net, output_layers, h, w, frame, colors, labels, FLAGS
        )


        if len(boxes) > 0:

            for i in range(len(boxes)):

                label = labels[classids[i]].lower().strip()

                x, y, bw, bh = boxes[i]

                confidence = confs[i]

                total_conf += confidence
                total_detections += 1


                if label in REAL_WIDTHS:

                    real_width = REAL_WIDTHS[label]

                    distance = (real_width * focal_length) / bw

                    distance_m = distance / 100


                    cv2.putText(
                        frame,
                        f"{label} {confidence:.2f} | {distance_m:.2f} m",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )


                    if ENABLE_SPEECH and time.time() - last_speech > 2:

                        speak(f"{label} at {distance_m:.1f} meters", engine)

                        last_speech = time.time()


        # ---------- PERFORMANCE METRICS ----------

        end_time = time.time()

        fps = 1 / (end_time - start_time)


        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent


        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"CPU: {cpu}%", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(frame, f"RAM: {ram}%", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        cv2.imshow(
            "Object Detection and Audio Assistance for Visually Impaired",
            frame
        )


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()


    # -------- FINAL METRICS --------

    if total_detections > 0:

        avg_conf = total_conf / total_detections

        print("\n=========== SYSTEM METRICS ===========")

        print(f"Total detections: {total_detections}")

        print(f"Average confidence: {avg_conf:.3f}")

        print("=======================================")


if __name__ == "__main__":
    main()