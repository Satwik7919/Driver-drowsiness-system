from flask import Flask, render_template, Response
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import cv2
import numpy as np
import dlib
import threading

app = Flask(__name__)
video_feed_running = True

mixer.init()
mixer.music.load("beep1.mp3")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 25
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
frame_count = 0

def driver_drowsiness():
    global cap, frame_count, video_feed_running
    while video_feed_running:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Optional: flip the frame horizontally
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 255), 1)

            if ear < thresh:
                frame_count += 1
                print(frame_count)
                if frame_count >= frame_check:
                    cv2.putText(frame, f"Frame Count: {frame_count}", (10, 60),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    cv2.putText(frame, "wake up", (10, 30),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    cv2.putText(frame, "wake up", (10, 325),
                                cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                    mixer.music.play()
            else:
                frame_count = 0
                cv2.putText(frame, f"Frame Count: {frame_count}", (10, 60),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                cv2.putText(frame, "Nice Driving", (10, 30),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('webpage.html')

@app.route('/video_feed')
def video_feed():
    return Response(driver_drowsiness(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    t = threading.Thread(target=driver_drowsiness)
    t.daemon = True
    t.start()
    app.run(debug=True)
