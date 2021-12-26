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
#  q
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
#
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
#         raise
#     webcam(page)
#     st.warning("Camera might take time to load! Please be patient!")
# except:
#     pass
#
#
#
#

import logging
import logging.handlers
import queue
import urllib.request
from pathlib import Path
from typing import Literal

import av
import cv2
import numpy as np
import PIL
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    ClientSettings,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent


@st.cache
def setup_logger():
    logging.basicConfig(level=logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)7s in %(module)s (%(filename)s:%(lineno)d): "
        "%(message)s"
    )
    ch.setFormatter(formatter)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")

    st_webrtc_logger.addHandler(ch)
    st_webrtc_logger.setLevel(logging.DEBUG)

    # `aiortc` does not have loggers with a common prefix
    # and the loggers cannot be configured in this way.
    # See https://github.com/aiortc/aiortc/issues/446
    # aiortc_logger = logging.getLogger("aiortc")
    # aiortc_logger.addHandler(ch)
    # aiortc_logger.setLevel(logging.DEBUG)

    logger = logging.getLogger(__name__)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main():
    st.header("WebRTC demo")

    object_detection_page = "Real time object detection (sendrecv)"
    video_filters_page = (
        "Real time video transform with simple OpenCV filters (sendrecv)"
    )
    streaming_page = (
        "Consuming media files on server-side and streaming it to browser (recvonly)"
    )
    sendonly_page = "WebRTC is sendonly and images are shown via st.image() (sendonly)"
    loopback_page = "Simple video loopback (sendrecv)"
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        [
            object_detection_page,
            video_filters_page,
            streaming_page,
            sendonly_page,
            loopback_page,
        ],
    )
    st.subheader(app_mode)

    if app_mode == video_filters_page:
        app_video_filters()
    elif app_mode == object_detection_page:
        app_object_detection()
    elif app_mode == streaming_page:
        app_streaming()
    elif app_mode == sendonly_page:
        app_sendonly()
    elif app_mode == loopback_page:
        app_loopback()


def app_loopback():
    """ Simple video loopback """
    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_class=None,  # NoOp
    )


def app_video_filters():
    """ Video transforms with OpenCV """

    class OpenCVVideoTransformer(VideoTransformerBase):
        type: Literal["noop", "cartoon", "edges", "rotate"]

        def __init__(self) -> None:
            self.type = "noop"

        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.type == "noop":
                pass
            elif self.type == "cartoon":
                # prepare color
                img_color = cv2.pyrDown(cv2.pyrDown(img))
                for _ in range(6):
                    img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
                img_color = cv2.pyrUp(cv2.pyrUp(img_color))

                # prepare edges
                img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img_edges = cv2.adaptiveThreshold(
                    cv2.medianBlur(img_edges, 7),
                    255,
                    cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY,
                    9,
                    2,
                )
                img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

                # combine color and edges
                img = cv2.bitwise_and(img_color, img_edges)
            elif self.type == "edges":
                # perform edge detection
                img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
            elif self.type == "rotate":
                # rotate image
                rows, cols, _ = img.shape
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
                img = cv2.warpAffine(img, M, (cols, rows))

            return img

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_class=OpenCVVideoTransformer,
        async_transform=True,
    )

    transform_type = st.radio(
        "Select transform type", ("noop", "cartoon", "edges", "rotate")
    )
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.type = transform_type

    st.markdown(
        "This demo is based on "
        "https://github.com/aiortc/aiortc/blob/2362e6d1f0c730a0f8c387bbea76546775ad2fe8/examples/server/server.py#L34. "
        "Many thanks to the project."
    )


def app_object_detection():
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """
    MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
    MODEL_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.caffemodel"
    PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
    PROTOTXT_LOCAL_PATH = HERE / "./models/MobileNetSSD_deploy.prototxt.txt"

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class NNVideoTransformer(VideoTransformerBase):
        confidence_threshold: float

        def __init__(self) -> None:
            self._net = cv2.dnn.readNetFromCaffe(
                str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH)
            )
            self.confidence_threshold = 0.8

        def _annotate_image(self, image, detections):
            # loop over the detections
            (h, w) = image.shape[:2]
            labels = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    # extract the index of the class label from the `detections`,
                    # then compute the (x, y)-coordinates of the bounding box for
                    # the object
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # display the prediction
                    label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
                    labels.append(label)
                    cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        image,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        COLORS[idx],
                        2,
                    )
            return image, labels

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            image = frame.to_ndarray(format="bgr24")
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
            )
            self._net.setInput(blob)
            detections = self._net.forward()
            annotated_image, labels = self._annotate_image(image, detections)
            # TODO: Show labels

            return annotated_image

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_transformer_class=NNVideoTransformer,
        async_transform=True,
    )

    confidence_threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
    )
    if webrtc_ctx.video_transformer:
        webrtc_ctx.video_transformer.confidence_threshold = confidence_threshold

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


def app_streaming():
    """ Media streamings """
    MEDIAFILES = {
        "big_buck_bunny_720p_2mb.mp4": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_2mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_2mb.mp4",
            "type": "video",
        },
        "big_buck_bunny_720p_10mb.mp4": {
            "url": "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",  # noqa: E501
            "local_file_path": HERE / "data/big_buck_bunny_720p_10mb.mp4",
            "type": "video",
        },
        "file_example_MP3_700KB.mp3": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_700KB.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_700KB.mp3",
            "type": "audio",
        },
        "file_example_MP3_5MG.mp3": {
            "url": "https://file-examples-com.github.io/uploads/2017/11/file_example_MP3_5MG.mp3",  # noqa: E501
            "local_file_path": HERE / "data/file_example_MP3_5MG.mp3",
            "type": "audio",
        },
    }
    media_file_label = st.radio(
        "Select a media file to stream", tuple(MEDIAFILES.keys())
    )
    media_file_info = MEDIAFILES[media_file_label]
    download_file(media_file_info["url"], media_file_info["local_file_path"])

    def create_player():
        return MediaPlayer(str(media_file_info["local_file_path"]))

        # NOTE: To stream the video from webcam, use the code below.
        # return MediaPlayer(
        #     "1:none",
        #     format="avfoundation",
        #     options={"framerate": "30", "video_size": "1280x720"},
        # )

    WEBRTC_CLIENT_SETTINGS.update(
        {
            "fmedia_stream_constraints": {
                "video": media_file_info["type"] == "video",
                "audio": media_file_info["type"] == "audio",
            }
        }
    )

    webrtc_streamer(
        key=f"media-streaming-{media_file_label}",
        mode=WebRtcMode.RECVONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        player_factory=create_player,
    )


def app_sendonly():
    """A sample to use WebRTC in sendonly mode to transfer frames
    from the browser to the server and to render frames via `st.image`."""
    webrtc_ctx = webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDONLY,
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )

    if webrtc_ctx.video_receiver:
        image_loc = st.empty()
        while True:
            try:
                frame = webrtc_ctx.video_receiver.frames_queue.get(timeout=1)
            except queue.Empty:
                print("Queue is empty. Stop the loop.")
                webrtc_ctx.video_receiver.stop()
                break

            img = frame.to_ndarray(format="bgr24")
            img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            image_loc.image(img)


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)

if __name__ == "__main__":
    setup_logger()
    main()