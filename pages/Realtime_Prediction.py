import streamlit as st
st.set_page_config(
    page_title="Person Recognition System | SixthSens AI",
    page_icon=":smiley:",
    layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
            .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
            .viewerBadge_text__1JaDK {display: none;}
            MainMenu {visibility: hidden;}
            header { visibility: hidden; }
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import face_rec
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import av


st.subheader("Face Recognition and Liveliness Detection System")

with st.spinner("Retriving data from DB"):
    redis_face_db = face_rec.retrive_data(name="academy:register")

st.success("Data retrived successfully")


realtimepred = face_rec.RealTimePred()  # realtime predict class


def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")  # 3 dimensional array

    pred_img = realtimepred.face_prediction(
        img, redis_face_db, feature_column="facial_features", thresh=0.5
    )

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(
    key="realtimePrediction",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [
        {
            "urls": "turn:a.relay.metered.ca:80?transport=tcp",
            "username": "9d79830e9a30b210d0582c23",
            "credential": "6TzT7r9tBsdKHdMD",
        },
        {
            "urls": "turn:a.relay.metered.ca:443?transport=tcp",
            "username": "9d79830e9a30b210d0582c23",
            "credential": "6TzT7r9tBsdKHdMD",
        },

    ]},
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, muted=True)

)
