import streamlit as st
import requests

API_URL = "http://localhost:8000"  # By default, FastAPI runs on port 8000

st.set_page_config(page_title="Review Rating Prediction")
st.title("Amazon Electronics Reviews -- Rating Prediction")
st.write(
    "Enter your review in English , the API will clean it and predict a rating from 1 to 5."
)

review_text = st.text_area("Review text in English")

if st.button("Predict rating"):  # If the button is clicked...
    if not review_text.strip():  # ...check if the input is empty
        st.warning("Please enter a review first.")
    else:  # ... if the text_area is not empty, proceed to call the API
        try:
            predicted_grade = requests.post(
                f"{API_URL}/predict", json={"review": review_text}
            )  # Call the /predict endpoint of the FastAPI app

            if predicted_grade.ok:  # If the response status code is 200 (OK)
                predicted = predicted_grade.json().get(
                    "predicted_grade", None
                )  # Extract the predicted
                # grade from the JSON response
                if predicted is not None:  # If a predicted grade was returned
                    st.success(f"Predicted Rating: {predicted} out of 5")

        except Exception as e:  # Catch any exceptions that occur during the API call
            st.error(f"An error occurred while calling the API: {e}")
