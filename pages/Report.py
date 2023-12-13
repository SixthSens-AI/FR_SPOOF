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

# # st.set_page_config(page_title="Reporting")
# st.subheader("Report")



st.header("Registered People")


if st.button('Refresh Data'):
    with st.spinner("Retriving data from Redis"):
        redis_face_db = face_rec.retrive_data(name="academy:register")
        # st.dataframe(redis_face_db)
        st.dataframe(redis_face_db[['name']])

