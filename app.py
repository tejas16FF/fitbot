import streamlit as st

st.set_page_config(page_title="FitBot - Week 1", page_icon="💪")
st.title("💪 FitBot - AI Powered Fitness Chatbot")
st.write("This is a **Week 1 demo** version. Currently, it shows dummy responses.")

user_input = st.text_input("Ask me about fitness, workouts, or diet:")

if user_input:
    if "workout" in user_input.lower():
        st.success("You should try 30 minutes of cardio and some light strength training 💪")
    elif "diet" in user_input.lower():
        st.success("Eat more protein and veggies while staying hydrated 🥗💧")
    else:
        st.success("Thanks for your question! The AI-powered responses will be added soon 🤖")
else:
    st.info("Enter a question above to see a sample response.")
