import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# mediapipe初期設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


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


# 姿勢のランドマーク
def pose(results, annotated_image, label, csv):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS)

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
def landmark(image, multi_landmarks):
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    annotated_image = image.copy()

    label = []
    csv = []

    label, csv = r_hand(results, annotated_image, label, csv)
    label, csv = pose(results, annotated_image, label, csv)

    # 全フレームのランドマークを結合する
    multi_landmarks.append(csv)
    return annotated_image, multi_landmarks


def images2movie(images):
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('ImgVideo.mp4', fourcc, 20.0, (height, width))

    for img in images:
        video.write(img)
    return video


base_dir = "/Users/shu/Desktop/"
model = tf.keras.models.load_model('weight.hdf5')
multi_landmarks = [[]]
landmark_images = []

st.title("リアルタイム手話認識")
uploaded_file = st.file_uploader("ファイルアップロード")

if uploaded_file is not None:
    uploaded_file = base_dir+uploaded_file.name
    st.video(uploaded_file)
    cap = cv2.VideoCapture(uploaded_file)

    if (cap.isOpened() == False):
        print("ビデオファイルを開くとエラーが発生しました")

    pred = np.zeros(20)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("Video", frame)

            # フレームにmediapipeを適用する
            annotated_image, multi_landmarks = landmark(
                frame, multi_landmarks)

            landmark_images.append(annotated_image)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # TODO:streamlit上で画像から動画を作成する
    # # annotated_imageを動画にする
    # landmark_vides = images2movie(landmark_images)
    # st.video(data=landmark_vides, format="TIFF")

    # multi_landmarksの末尾60フレームで識別する
    if len(multi_landmarks) > 60:
        array_landmark = np.array(multi_landmarks[-60:])
        array_landmark = np.nan_to_num(array_landmark, nan=0.1)

        pred = model.predict(array_landmark[None, ...])
    st.write("sign"+str(pred.argmax()+1) +
             "である確率は" + str(int(pred.max()*100))+"％")
