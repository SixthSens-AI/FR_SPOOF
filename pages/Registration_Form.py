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
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
import av
import cv2

st.subheader("Registration Form")

## Init registration form
registration_form = face_rec.RegistrationForm()

# 1. collect person information
col1, col2 = st.columns(2)
name = col1.text_input("Name", placeholder="Enter your name")

pic_option = col2.radio('Upload Picture', options=["Upload a Picture", "Click a picture"])

# 2. collect facial embedding
def video_callback_func(frame):
    img = frame.to_ndarray(format="bgr24")  # 3 dimensional array
    reg_img, embedding = registration_form.get_embedding(img)

    if embedding is not None:
        with open("face_embedding.txt", mode="ab") as f:
            np.savetxt(f, embedding)

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

if pic_option == 'Upload a Picture':
    # Add a file uploader
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)  # Convert to a numpy array
            reg_img, embedding = registration_form.get_embedding(img)
            if embedding is not None:
                with open("face_embedding.txt", mode="ab") as f:
                    np.savetxt(f, embedding)


elif pic_option == 'Click a picture':
    webrtc_streamer(
        key="registration",
        video_frame_callback=video_callback_func,
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

if st.button("Submit"):
    return_val = registration_form.save_data_in_redis_db(name)
    if return_val == True:
        st.success("Data saved successfully")
    else:
        st.error("Something went wrong")
