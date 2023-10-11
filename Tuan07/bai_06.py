import streamlit as st
import time

st.balloons()

st.subheader("Progess bar")
st.progress(10)

st.subheader("Wait the execution")
with st.spinner('Wait for it...'):    
	time.sleep(10)