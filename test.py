
# import logging
# import threading
# from pathlib import Path
import mediapipe as mp

# try:
#     from typing import Literal
# except ImportError:
#     from typing_extensions import Literal  # type: ignore

import av
import cv2
import streamlit as st

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# HERE = Path(__file__).parent
#
# logger = logging.getLogger(__name__)


mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


# def main():
#     st.header("WebRTC demo")
#     video_filters_page = (
#         "Real time video transform with simple OpenCV filters (sendrecv)"
#     )
#
#     app_mode = st.sidebar.selectbox(
#         "Choose the app mode",
#         [
#             video_filters_page],
#     )
#     st.subheader(app_mode)
#
#     if app_mode == video_filters_page:
#         app_video_filters()


    # logger.debug("=== Alive threads ===")
    # for thread in threading.enumerate():
    #     if thread.is_alive():
    #         logger.debug(f"  {thread.name} ({thread.ident})")


def app_video_filters():
    """ Video transforms with OpenCV """

    class OpenCVVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

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

    # if webrtc_ctx.video_processor:
    #     webrtc_ctx.video_processor.type = st.radio(
    #         "Select transform type", ("noop", "cartoon", "edges", "rotate")
    #     )



# if __name__ == "__main__":
#     import os
#
#     DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]
#
#     logging.basicConfig(
#         format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
#         "%(message)s",
#         force=True,
#     )
#
#     logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)
#
#     st_webrtc_logger = logging.getLogger("streamlit_webrtc")
#     st_webrtc_logger.setLevel(logging.DEBUG)
#
#     fsevents_logger = logging.getLogger("fsevents")
#     fsevents_logger.setLevel(logging.WARNING)
#
app_video_filters()