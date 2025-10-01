import streamlit as st
import pandas as pd
from transformers import pipeline
import json


def read_txt(file) -> str:
    return file.read().decode("utf-8")

def isolate_patient_dialogue(transcript: str) -> list:
    lines = transcript.splitlines()
    for line in lines:
        cleaned_line = line.strip()
        # Look for lines that start with "Patient:"
        if cleaned_line.lower().startswith("patient:"):
            # Extract the text after "Patient:"
            utterance = cleaned_line[len("patient:"):].strip()
            if utterance:
                patient_lines.append(utterance)
    return patient_lines


st.set_page_config(page_title="Patient Sentiment Analyzer", layout="centered")
st.title("🩺 Patient Sentiment & Intent Analyzer")
st.markdown("Upload a clinical transcript (.txt) to analyze the sentiment of the patient's dialogue.")

try:
    # Use the correct relative path to the model
    model_path = "../utils/final_sentiment_model"
    sentiment_classifier = pipeline("text-classification", model=model_path)
    st.sidebar.success("Fine-tuned sentiment model loaded successfully!")
except Exception as e:
    st.sidebar.error("Could not load local fine-tuned model. Ensure you have trained it and the folder './final_sentiment_model' exists.")
    st.sidebar.caption(f"Error: {e}")
    sentiment_classifier = None
try:
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    st.sidebar.success("Zero-shot intent model loaded successfully!")
except Exception as e:
    st.sidebar.error("Could not load Zero-Shot model from Hugging Face.")
    intent_classifier = None


uploaded_file = st.file_uploader("Upload your transcript file", type=["txt"])

if uploaded_file is not None:
    transcript = read_txt(uploaded_file)
    
    with st.expander("Show Full Transcript", expanded=False):
        st.text_area("", transcript, height=250)
    
    patient_dialogue = isolate_patient_dialogue(transcript)

    if not patient_dialogue:
        st.warning("No patient dialogue found. Please ensure lines start with 'Patient:'.")
    else:
        st.subheader("Patient Dialogue Analysis")
        
        results = []
        if sentiment_classifier and intent_classifier:
            with st.spinner("Analyzing patient dialogue..."):
                for line in patient_dialogue:
                    analysis = {}
                    analysis["text"] = line
                    
                    sentiment_result = sentiment_classifier(line)[0]
                    analysis["sentiment"] = sentiment_result['label']
                    analysis["sentiment_score"] = round(sentiment_result['score'], 2)
                    
                    intent_labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern", "Expressing gratitude", "Reporting outcome"]
                    intent_result = intent_classifier(line, candidate_labels=intent_labels)
                    analysis["intent"] = intent_result['labels'][0]
                    analysis["intent_score"] = round(intent_result['scores'][0], 2)
                    
                    results.append(analysis)
            
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            st.subheader("JSON Output")
            json_output = json.dumps(results, indent=2)
            st.code(json_output, language="json")
            st.download_button(
                label="Download JSON",
                data=json_output,
                file_name="patient_sentiment_analysis.json",
                mime="application/json",
            )
        else:
            st.error("One or more AI models could not be loaded. Please check the sidebar for errors.")