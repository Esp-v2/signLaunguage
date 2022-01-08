import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
#from PIL import Image
import numpy as np
#import pandas as pd
import tensorflow as tf

# 参考
# streamlit https://zenn.dev/whitphx/articles/streamlit-realtime-cv-app
###


# mediapipe初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# 顔のランドマーク
def face(results, annotated_image, label, csv):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        for index, landmark in enumerate(results.face_landmarks.landmark):
            label.append("face_"+str(index) + "_x")
            label.append("face_"+str(index) + "_y")
            label.append("face_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)
    else:
        for index in range(468):
            label.append("face_"+str(index) + "_x")
            label.append("face_"+str(index) + "_y")
            label.append("face_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)
    return label, csv


# 右手のランドマーク
def r_hand(results, annotated_image, label, csv):
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)

        for index, landmark in enumerate(results.right_hand_landmarks.landmark):
            label.append("r_hand_"+str(index) + "_x")
            label.append("r_hand_"+str(index) + "_y")
            label.append("r_hand_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)
    else:
        for index in range(21):
            label.append("r_hand_"+str(index) + "_x")
            label.append("r_hand_"+str(index) + "_y")
            label.append("r_hand_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)
    return label, csv


# 左手のランドマーク
def l_hand(results, annotated_image, label, csv):
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)

        for index, landmark in enumerate(results.left_hand_landmarks.landmark):
            label.append("l_hand_"+str(index) + "_x")
            label.append("l_hand_"+str(index) + "_y")
            label.append("l_hand_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)
    else:
        for index in range(21):
            label.append("l_hand_"+str(index) + "_x")
            label.append("l_hand_"+str(index) + "_y")
            label.append("l_hand_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)
    return label, csv


# 姿勢のランドマーク
def pose(results, annotated_image, label, csv):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS)

        for index, landmark in enumerate(results.pose_landmarks.landmark):
            label.append("pose_"+str(index) + "_x")
            label.append("pose_"+str(index) + "_y")
            label.append("pose_"+str(index) + "_z")
            csv.append(landmark.x)
            csv.append(landmark.y)
            csv.append(landmark.z)
    else:
        for index in range(33):
            label.append("pose_"+str(index) + "_x")
            label.append("pose_"+str(index) + "_y")
            label.append("pose_"+str(index) + "_z")
            for _ in range(3):
                csv.append(np.nan)
    return label, csv


# mediapipeでランドマークを出力する
def landmark(image):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    label = []
    csv = []

    #label, csv = face(results, annotated_image, csv, label)
    label, csv = r_hand(results, annotated_image, label, csv)
    #label, csv = l_hand(results, annotated_image, csv, label)
    label, csv = pose(results, annotated_image, label, csv)

    # 全フレームのランドマークを結合する
    multi_landmarks.append(csv)

    # multi_landmarksの末尾60フレームで識別する
    if len(multi_landmarks) > 60:
        array_landmark = np.array(multi_landmarks[-60:])
        array_landmark = np.nan_to_num(array_landmark, nan=0.1)

        pred = model.predict(array_landmark[None, ...])
        print(pred.argmax())

    return annotated_image


class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = landmark(img)  # mediapipeでのランドマーク検出

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.title("リアルタイム手話認識")

multi_landmarks = [[]]
model = tf.keras.models.load_model('LSTM.hdf5')
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
