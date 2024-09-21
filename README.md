# Retail_theft_detection
import cv2
import numpy as np
import time
from twilio.rest import Client

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

account_sid = 'AC2ec26b2aaa09375b44451c7f536db209'
auth_token = 'b167a98338167964eaccd2fbb3e35011'
client = Client(account_sid, auth_token)

alert_sent = False
alert_delay = 15
def send_alert():
    global alert_sent
    if not alert_sent:
        time.sleep(alert_delay)
        message = client.messages.create(
            body="Alert:Potential theft detected!!",
            from_='+14014062674',  # Your Twilio number
            to='+918619990828'  # Your phone number
    )
        print(f"Alert sent: {message.sid}")
    alert_sent = True

# Define a function to detect objects
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], class_ids[i]) for i in indexes]

# Initialize video capture
cap = cv2.VideoCapture("WhatsApp_cosmetics.mp4")

# Initialize trackers
item_trackers = []
customer_trackers = []
theft_detected = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)

    for (box, class_id) in detections:
        x, y, w, h = box
        label = str(classes[class_id])
        if label == "person":
            tracker = cv2.TrackerKCF_create()
            customer_trackers.append(tracker)
            tracker.init(frame, tuple(box))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle for person

        elif label in ["handbag", "bottle"]:
            tracker = cv2.TrackerKCF_create()
            item_trackers.append(tracker)
            tracker.init(frame, tuple(box))

        success, person_box = tracker.update(frame)
        if success:
            px, py, pw, ph = map(int, person_box)
            for item_tracker in item_trackers:
                success, item_box = item_tracker.update(frame)
                if success:
                    ix, iy, iw, ih = map(int, item_box)
                    if px < ix < px + pw and py < iy < py + ph:
                        theft_detected = True
                        send_alert()
                        alert_send = True
                        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)  # Red rectangle for theft
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
