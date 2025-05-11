import argparse

import numpy as np
import cv2 as cv
import joblib

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image1', '-i1', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--image2', '-i2', type=str, help='Path to the input image2. When image1 and image2 parameters given then the program try to find a face on both images and runs face recognition algorithm.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='../model/face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='../model/face_recognition_sface_2021dec.onnx', help='Path to the face recognition model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', type=str2bool, default=False, help='Set true to save results. This flag is invalid when using camera.')
args = parser.parse_args()

svc = joblib.load('../model/svc.pkl')
mydict = ['thayDuong', 'thayKien', 'thayNguyen','thayTin','thayTruong']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for face in faces[1]:
            # Căn chỉnh và trích xuất đặc trưng khuôn mặt
            face_align = recognizer.alignCrop(frame, face)
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]

            # Vẽ khung chữ nhật quanh khuôn mặt
            coords = face[:4].astype(np.int32)  # (x, y, w, h)
            x, y, w, h = coords
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Ghi tên ở phía trên khung
            text_size, _ = cv.getTextSize(result, cv.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            text_w, text_h = text_size
            cv.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), (0, 255, 0), -1)
            cv.putText(frame, result, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)



if __name__ == '__main__':
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    recognizer = cv.FaceRecognizerSF.create(
    args.face_recognition_model,"")

    tm = cv.TickMeter()

    cap = cv.VideoCapture(0)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        
        key = cv.waitKey(1)
        if key == 27:
            break
        if faces[1] is not None:
            face_align = recognizer.alignCrop(frame, faces[1][0])
            face_feature = recognizer.feature(face_align)
            test_predict = svc.predict(face_feature)
            result = mydict[test_predict[0]]
            cv.putText(frame,result,(1,50),cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        cv.imshow('Live', frame)
    cv.destroyAllWindows()
