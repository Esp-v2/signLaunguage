import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
from PIL import Image
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
#mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


st.title("リアルタイム手話認識")

###
#  参考 https://zenn.dev/whitphx/articles/streamlit-realtime-cv-app
###
class VideoProcessor:
    def landmark(self,image):
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=2,
            min_detection_confidence=0.5) as face_mesh:
            # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Draw face landmarks of each face.
            annotated_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
            return annotated_image

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        #img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)  # エッジ検出
        img = self.landmark(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
