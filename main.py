import streamlit as st

st.title("Tensorflow Playgroud by Aravinth")
st.page_link("main.py", label="Home", icon="🏠")

st.subheader("Welcome to Tensorflow Playground")

st.markdown('''1. You can create a :red[Random Dataset] and train your :blue[Customized Neural Network] on the Randomly Generated Data.''')
st.page_link("pages/Create_Custom_Dataset.py", label=":green[click here]")

st.markdown('''2. This Feature provide :red[Nine] different Dataset You can use them and play with you :blue[customized Neural Network]''')
st.page_link("pages/Select_Existed_Dataset.py", label=":green[click here]")
st.sidebar.header("")