# import streamlit as st
# import mediapipe as mp
# import cv2
# # from deepface import DeepFace
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#
# # st.set_page_config(layout="wide")
# col = st.empty()
# #remove
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
#
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils
#
# noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
# mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
# from streamlit_webrtc import (
#     AudioProcessorBase,
#     RTCConfiguration,
#     VideoProcessorBase,
#     WebRtcMode,
#     webrtc_streamer,
# )
#
# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal  # type: ignore
# import av
#
# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )

# def webcam(page):
#
#     class OpenCVVideoProcessor(VideoProcessorBase):
#         def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#             img = frame.to_ndarray(format="bgr24")
#
#             st.warning("Click R to Start Cam!")
#             if page == "handDetector":
#                 imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 results = hands.process(imgRGB)
#
#                 if results.multi_hand_landmarks:
#                     for handLms in results.multi_hand_landmarks:
#                         mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#             elif page == "poseDetector":
#                 imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 results = pose.process(imgRGB)
#                 mpDraw.draw_landmarks(
#                     img,
#                     results.pose_landmarks,
#                     mpPose.POSE_CONNECTIONS)
#
#             # elif page == "facialSentimentAnalysis":
#             #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
#             #     faces = faceCascade.detectMultiScale(gray, 1.1, 4)
#             #     cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
#             #     try:
#             #         predictions = DeepFace.analyze(img, actions=['emotion'])
#             #         for (x, y, w, h) in faces:
#             #             cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             #         cv2.putText(img, predictions['dominant_emotion'], (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#             #     except:
#             #         cv2.putText(img, "No Face Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#             #     return av.VideoFrame.from_ndarray(img, format="bgr24")
#
#             elif page == "handSignRecognizer":
#                 imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 results = hands.process(imgRGB)
#                 cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
#                 if results.multi_hand_landmarks:
#                     # print(results.multi_hand_landmarks)
#                     for handLms in results.multi_hand_landmarks:
#                         # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#                         lmList = []
#                         myHand = results.multi_hand_landmarks[0]
#                         for id, lm in enumerate(myHand.landmark):
#                             # print(id, lm)
#                             h, w, c = img.shape
#                             cx, cy = int(lm.x * w), int(lm.y * h)
#                             # print(id, cx, cy)
#                             lmList.append([id, cx, cy])
#                         # print(lmList)
#                         if len(lmList) != 0:
#                             print(lmList)
#                             if lmList[8][2] < lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
#                                     lmList[14][
#                                         2] and lmList[20][2] > lmList[18][2]:
#                                 cv2.putText(img, "Pointing Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] > \
#                                     lmList[14][
#                                         2] and lmList[20][2] > lmList[18][2]:
#                                 cv2.putText(img, "Peace Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] < lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
#                                     lmList[14][
#                                         2] and lmList[20][2] < lmList[18][2]:
#                                 cv2.putText(img, "Rock Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] > lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
#                                     lmList[14][
#                                         2] and lmList[20][2] > lmList[18][2] and lmList[4][2] < lmList[3][2]:
#                                 cv2.putText(img, "Thumbs Up Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] > lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] > \
#                                     lmList[14][
#                                         2] and lmList[20][2] > lmList[18][2]:
#                                 cv2.putText(img, "Obscene Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] > lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] < \
#                                     lmList[14][
#                                         2] and lmList[20][2] < lmList[18][2] and lmList[4][2] < lmList[3][2]:
#                                 cv2.putText(img, "OK Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] > lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
#                                     lmList[14][
#                                         2] and lmList[20][2] > lmList[18][2]:
#                                 cv2.putText(img, "Fist Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             elif lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] < \
#                                     lmList[14][
#                                         2] and lmList[20][2] < lmList[18][2] and lmList[4][2] <= lmList[3][2]:
#                                 cv2.putText(img, "High Five Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                             else:
#                                 cv2.putText(img, "Hand Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                         else:
#                             cv2.putText(img, "No Hand Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                 elif page == "maskDetector":
#                     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                     nose = noseCascade.detectMultiScale(gray, 1.3, 5)
#                     mouth = mouthCascade.detectMultiScale(gray, 1.3, 5)
#                     cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
#                     if len(nose) != 0 and len(mouth) != 0:
#                         cv2.putText(img, "Not Wearing Mask", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#                     else:
#                         cv2.putText(img, "Good, Wearing Mask!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
#
#                 elif page == "sketchYourself":
#                     option = st.selectbox(
#                         'Select Sketch Type',
#                         ('', 'Cartoon', 'Edges', 'Colored'))
#                     if option == "":
#                         pass
#
#                     elif option == "Cartoon":
#                         # prepare color
#                         img_color = cv2.pyrDown(cv2.pyrDown(img))
#                         for _ in range(6):
#                             img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
#                         img_color = cv2.pyrUp(cv2.pyrUp(img_color))
#
#                         # prepare edges
#                         img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#                         img_edges = cv2.adaptiveThreshold(
#                             cv2.medianBlur(img_edges, 7),
#                             255,
#                             cv2.ADAPTIVE_THRESH_MEAN_C,
#                             cv2.THRESH_BINARY,
#                             9,
#                             2,
#                         )
#                         img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
#
#                         # combine color and edges
#                         img = cv2.bitwise_and(img_color, img_edges)
#                     elif option == "Edges":
#                         # perform edge detection
#                         img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
#
#                     elif option == "Colored":
#                         # convert to RGB
#                         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                         # Sobel Edges
#                         x_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
#                         y_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
#                         img = cv2.bitwise_or(x_sobel, y_sobel)
#
#             return av.VideoFrame.from_ndarray(img, format="bgr24")
#
#     # webrtc_ctx = webrtc_streamer(
#     #     key="opencv-filter",
#     #     mode=WebRtcMode.SENDRECV,
#     #     rtc_configuration=RTC_CONFIGURATION,
#     #     video_processor_factory=OpenCVVideoProcessor,
#     #     media_stream_constraints={"video": True, "audio": False},
#     #     async_processing=True,
#     # )
#     webrtc_ctx = webrtc_streamer(
#         key="opencv-filter",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,
#         video_processor_factory=OpenCVVideoProcessor,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#         video_html_attrs={
#             "style": {"margin": "0 auto", "border": "5px yellow solid"},
#             "controls": False,
#             "autoPlay": True,
#         },
#     )
#
#
# page = ""
# pages = ['handDetector','maskDetector','poseDetector','handSignRecognizer','facialSentimentAnalysis','sketchYourself']
# try:
#     query_params = st.experimental_get_query_params()
#     page = query_params['page'][0]
#     if not page in pages:
#         st.header("404 Error! Page Not Found!")
#
#     webcam(page)
#     st.warning("Camera might take time to load! Please be patient!")
# except:
#     pass






import streamlit as st
import mediapipe as mp
import cv2
# from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# st.set_page_config(layout="wide")
col = st.empty()
#remove
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
import av

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

noseCascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
mouthCascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

st.write("Press start to turn on Camera!")
st.write("If camera doesn't turn on, click the select device button, change the camera input and reload your screen!")




def handDetector():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
def maskDetector():
    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            nose = noseCascade.detectMultiScale(gray, 1.3, 5)
            mouth = mouthCascade.detectMultiScale(gray, 1.3, 5)
            cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
            if len(nose) != 0 and len(mouth) != 0:
                cv2.putText(img, "Not Wearing Mask", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
            else:
                cv2.putText(img, "Good, Wearing Mask!", (20, 50), cv2.FONT_HERSHEY_DUPLEX,1, 255)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_html_attrs={
            "style": {"margin": "0 auto", "border": "5px yellow solid"},
            "controls": False,
            "autoPlay": True,
        },
    )

def app():
    page = ""
    pages = ['handDetector', 'maskDetector', 'poseDetector', 'handSignRecognizer', 'facialSentimentAnalysis',
             'sketchYourself']
    try:
        query_params = st.experimental_get_query_params()
        page = query_params['page'][0]
        if page == "handDetector":
            handDetector()
        elif page == "maskDetector":
            maskDetector()
        else:
            st.header("404 Error! Page Not Found!")


        st.warning("Camera might take time to load! Please be patient!")
    except:
        pass

if __name__ == "__main__":
    app()
