import streamlit as st
import mediapipe as mp
import cv2

# st.set_page_config(layout="wide")
col = st.empty()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

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

def webcam(page):

    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if page == "handDetector":
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)

                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            elif page == "handSignRecognizer":
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(imgRGB)
                cv2.rectangle(img, pt1=(0, 0), pt2=(700, 80), color=(0, 0, 0), thickness=-1)
                if results.multi_hand_landmarks:
                    # print(results.multi_hand_landmarks)
                    for handLms in results.multi_hand_landmarks:
                        # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                        lmList = []
                        myHand = results.multi_hand_landmarks[0]
                        for id, lm in enumerate(myHand.landmark):
                            # print(id, lm)
                            h, w, c = img.shape
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            # print(id, cx, cy)
                            lmList.append([id, cx, cy])
                        # print(lmList)
                        if len(lmList) != 0:
                            print(lmList)
                            if lmList[8][2] < lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
                                    lmList[14][
                                        2] and lmList[20][2] > lmList[18][2]:
                                cv2.putText(img, "Pointing Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] > \
                                    lmList[14][
                                        2] and lmList[20][2] > lmList[18][2]:
                                cv2.putText(img, "Peace Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] < lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
                                    lmList[14][
                                        2] and lmList[20][2] < lmList[18][2]:
                                cv2.putText(img, "Rock Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] > lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
                                    lmList[14][
                                        2] and lmList[20][2] > lmList[18][2] and lmList[4][2] < lmList[3][2]:
                                cv2.putText(img, "Thumbs Up Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] > lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] > \
                                    lmList[14][
                                        2] and lmList[20][2] > lmList[18][2]:
                                cv2.putText(img, "Obscene Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] > lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] < \
                                    lmList[14][
                                        2] and lmList[20][2] < lmList[18][2] and lmList[4][2] < lmList[3][2]:
                                cv2.putText(img, "OK Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] > lmList[6][2] and lmList[12][2] > lmList[10][2] and lmList[16][2] > \
                                    lmList[14][
                                        2] and lmList[20][2] > lmList[18][2]:
                                cv2.putText(img, "Fist Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            elif lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2] and lmList[16][2] < \
                                    lmList[14][
                                        2] and lmList[20][2] < lmList[18][2] and lmList[4][2] <= lmList[3][2]:
                                cv2.putText(img, "High Five Sign", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

                            else:
                                cv2.putText(img, "Hand Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)
                else:
                    cv2.putText(img, "No Hand Detected!", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, 255)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    # webrtc_ctx = webrtc_streamer(
    #     key="opencv-filter",
    #     mode=WebRtcMode.SENDRECV,
    #     rtc_configuration=RTC_CONFIGURATION,
    #     video_processor_factory=OpenCVVideoProcessor,
    #     media_stream_constraints={"video": True, "audio": False},
    #     async_processing=True,
    #     video_html_attrs={
    #         "style": {"margin": "0 auto", "border": "5px yellow solid"},
    #         "controls": False,
    #         "autoPlay": True,
    #     },
    # )


page = ""
pages = ['handDetector','maskDetector','poseDetector','handSignRecognizer','facialSentimentAnalysis','sketchYourself']
try:
    query_params = st.experimental_get_query_params()
    page = query_params['page'][0]
    if not page in pages:
        st.header("404 Error! Page Not Found!")
        raise
    webcam(page)
    st.warning("Camera might take time to load! Please be patient!")
except:
    pass




