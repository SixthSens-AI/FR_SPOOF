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



st.header("Person Recognition System")

with st.spinner("Loading model and connecting to DB"):
    import face_rec

st.success('Model loaded successfully')
st.success('Connected to DB')

