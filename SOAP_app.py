import streamlit as st
import google.generativeai as genai
import json
import os

# --- Core Hybrid Function ---
def generate_soap_note_hybrid(api_key, transcript_text):
    """
    Generates a SOAP note using a hybrid approach:
    1.  Prompts the Gemini LLM to create a JSON object with a specific nested structure.
    2.  Validates the LLM's output to ensure all required fields are present.
    """
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return {"Error": "Authentication failed. Please check your API Key.", "Details": str(e)}

    # Define the precise JSON structure we want the LLM to create.
    # The descriptions now emphasize brevity and conciseness.
    json_schema = """
    {
      "Subjective": {
        "Chief_Complaint": "Brief, to-the-point reason for the visit (e.g., 'Neck and back pain').",
        "History_of_Present_Illness": "Concise summary of the event, symptoms, and treatment. Be brief and factual."
      },
      "Objective": {
        "Physical_Exam": "Key findings from the physical exam. Be brief (e.g., 'Full range of motion, no tenderness').",
        "Observations": "Short, objective observations (e.g., 'Patient appears in normal health')."
      },
      "Assessment": {
        "Diagnosis": "The primary diagnosis (e.g., 'Whiplash injury').",
        "Severity": "One or two words describing severity (e.g., 'Mild, improving')."
      },
      "Plan": {
        "Treatment": "Concise treatment plan (e.g., 'Continue physiotherapy as needed').",
        "Follow_Up": "Brief follow-up instructions (e.g., 'Return if pain worsens')."
      }
    }
    """

    # Create the prompt for the Gemini model
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    
    prompt = f"""
    You are an expert medical scribe for a busy clinic. Your task is to convert the following transcript into a concise, to-the-point JSON object. 
    The output must be crisp and easy to read quickly. Avoid narrative explanations and use brief statements.

    Adhere strictly to this JSON schema and use the field descriptions as a guide for what to include:
    ---
    {json_schema}
    ---

    Transcript to analyze:
    ---
    {transcript_text}
    ---

    Generate ONLY the JSON object based on the transcript.
    """
    
    # --- 1. LLM Generation ---
    try:
        response = model.generate_content(prompt)
        # Clean up the output to be valid JSON
        cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "")
        llm_output = json.loads(cleaned_json_string)
    except json.JSONDecodeError:
        return {"Error": "LLM output was not valid JSON.", "RawOutput": response.text}
    except Exception as e:
        return {"Error": "Failed to generate content from the LLM.", "Details": str(e)}

    # --- 2. Rule-Based Validation ---
    # Define the complete, ideal structure of our final JSON object
    final_soap_note = {
        "Subjective": {
            "Chief_Complaint": "Not specified",
            "History_of_Present_Illness": "Not specified"
        },
        "Objective": {
            "Physical_Exam": "Not specified",
            "Observations": "Not specified"
        },
        "Assessment": {
            "Diagnosis": "Not specified",
            "Severity": "Not specified"
        },
        "Plan": {
            "Treatment": "Not specified",
            "Follow_Up": "Not specified"
        }
    }

    # Iterate through the expected structure and fill it with data from the LLM
    # This ensures that even if the LLM misses a field, our final output is consistent.
    for section, fields in final_soap_note.items():
        if section in llm_output and isinstance(llm_output[section], dict):
            for field, default_value in fields.items():
                # Use the LLM's value if it exists and is not empty, otherwise keep the default
                final_soap_note[section][field] = llm_output[section].get(field, default_value)

    return final_soap_note


# --- Streamlit User Interface ---

st.set_page_config(layout="wide")
st.title("🩺 Hybrid AI SOAP Note Generator")
st.markdown("This application uses a powerful LLM to generate content and rule-based validation to ensure a perfectly structured output.")

# 1. User Inputs
st.header("1. Enter API Key & Upload Transcript")

# Use a container for better layout
with st.container():
    api_key = st.text_input("Enter your Google AI Studio API Key", type="password", help="Your key is required to run the AI model.")
    uploaded_file = st.file_uploader("Choose a .txt transcript file", type="txt")

if 'soap_note_json' not in st.session_state:
    st.session_state.soap_note_json = ""

# 2. Display Transcript and Generate Button
if uploaded_file is not None:
    transcript_text = uploaded_file.getvalue().decode("utf-8")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Transcript")
        st.text_area("Content", transcript_text, height=400, key="transcript_display")
    
    if st.button("✨ Generate SOAP Note", use_container_width=True):
        if not api_key:
            st.error("A Google AI Studio API Key is required.")
        else:
            with st.spinner('AI is analyzing the transcript and structuring the note...'):
                soap_note_dict = generate_soap_note_hybrid(api_key, transcript_text)
                st.session_state.soap_note_json = json.dumps(soap_note_dict, indent=2)

    if st.session_state.soap_note_json:
        with col2:
            st.subheader("Generated SOAP Note")
            st.json(st.session_state.soap_note_json)
            
            st.download_button(
               label="Download SOAP Note as JSON",
               data=st.session_state.soap_note_json,
               file_name="soap_note.json",
               mime="application/json",
               use_container_width=True
            )

