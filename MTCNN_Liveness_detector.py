import cv2
import numpy as np
from facenet_pytorch.models.mtcnn import MTCNN
import torch
from tensorflow.keras.models import model_from_json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)
mtcnn = MTCNN(keep_all=True, device=device, selection_method='fastest')


# Load CNN Model graph
json_file = open('mobilenetold2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load CNN model
model.load_weights('mobilenetold2.h5')
print("Model loaded from disk")

def anti_spoofing(frame):
    # Convert BGR image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(rgb)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.astype(np.int32)
            face_roi = frame[y1:y2, x1:x2]
            resized_face = cv2.resize(face_roi, (160, 160))
            resized_face = resized_face.astype("float") / 255.0
            resized_face = np.expand_dims(resized_face, axis=0)
            preds = model.predict(resized_face)[0]
            if preds > 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                label = 'real'
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

video = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = video.read()
        frame = anti_spoofing(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        pass
video.release()
cv2.destroyAllWindows()
