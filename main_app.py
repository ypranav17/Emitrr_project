import streamlit as st
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import functools
import re
)
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.error("Transformers library not found. Please ensure it's installed correctly.")
    TRANSFORMERS_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

def read_txt(file) -> str:
    """Reads and decodes the uploaded text file."""
    return file.read().decode("utf-8")

def isolate_patient_dialogue(transcript: str) -> list:
    """Extracts only the lines spoken by the patient."""
    patient_lines = []
    lines = transcript.splitlines()
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line.lower().startswith("patient:"):
            utterance = cleaned_line[len("patient:"):].strip()
            if utterance:
                patient_lines.append(utterance)
    return patient_lines


@functools.lru_cache(maxsize=None)
def get_medical_ner_pipeline():
    if not TRANSFORMERS_AVAILABLE: return None
    return pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

@functools.lru_cache(maxsize=None)
def get_general_ner_pipeline():
    if not TRANSFORMERS_AVAILABLE: return None
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

@functools.lru_cache(maxsize=None)
def get_qa_pipeline():
    if not TRANSFORMERS_AVAILABLE: return None
    return pipeline("question-answering", model="deepset/roberta-large-squad2")

@functools.lru_cache(maxsize=None)
def get_sentiment_classifier():
    if not TRANSFORMERS_AVAILABLE: return None
    return pipeline("text-classification", model="../utils/final_sentiment_model")

@functools.lru_cache(maxsize=None)
def get_intent_classifier():
    if not TRANSFORMERS_AVAILABLE: return None
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def process_symptoms(entities: list, confidence_threshold: float = 0.70) -> list:
    SYMPTOM_STOP_WORDS = {"fell", "tripped", "motion", "objects", "move", "couldn't", "couldn"}
    high_confidence_entities = [e for e in entities if e['score'] > confidence_threshold]
    symptoms = [e for e in high_confidence_entities if e['entity_group'] == 'Sign_symptom']
    body_parts = [e for e in high_confidence_entities if e['entity_group'] == 'Biological_structure']
    combined_symptoms, used_body_parts = [], []
    for symp in symptoms:
        symp_word = symp['word'].replace("##", "").lower()
        if symp_word in SYMPTOM_STOP_WORDS: continue
        found_link = False
        for part in body_parts:
            if part not in used_body_parts and 0 < symp['start'] - part['end'] < 25:
                combined_symptoms.append(f"{part['word']} {symp_word}")
                used_body_parts.append(part)
                found_link = True
                break
        if not found_link: combined_symptoms.append(symp_word)
    return sorted(list(set(combined_symptoms)))

def extract_medical_info(text: str) -> dict:
    medical_ner_pl = get_medical_ner_pipeline()
    if not medical_ner_pl: return {}
    results = medical_ner_pl(text)
    processed_symptoms = process_symptoms(results)
    entity_map = {"Treatment": ["Medication", "Therapeutic_procedure", "Diagnostic_procedure"]}
    extracted = defaultdict(list)
    extracted["Symptoms"] = processed_symptoms
    for entity in results:
        entity_group, entity_word = entity['entity_group'], entity['word'].strip().replace("##", "")
        for key, labels in entity_map.items():
            if entity_group in labels and entity_word not in extracted[key]:
                extracted[key].append(entity_word)
    return dict(extracted)

def extract_patient_name(text: str) -> str | None:
    general_ner_pl = get_general_ner_pipeline()
    if not general_ner_pl: return None
    results = general_ner_pl(text)
    for entity in results:
        if entity['entity_group'] == 'PER': return entity['word']
    match = re.search(r"(?:Mr\.|Ms\.|Mrs\.)\s+[A-Za-z]+", text, re.IGNORECASE)
    if match: return match.group(0)
    return None

def extract_qa_info(context: str) -> dict:
    qa_pl = get_qa_pipeline()
    if not qa_pl: return {}
    questions = {
        "Diagnosis": "What was the patient diagnosed with?",
        "Current_Status": "What pain or symptoms is the patient still experiencing now?",
        "Prognosis": "What is the doctor's prognosis for the patient's recovery?"
    }
    extracted = {}
    for key, question in questions.items():
        result = qa_pl(question=question, context=context)
        if result['score'] > 0.05: extracted[key] = result['answer'].strip().rstrip('.')
    return extracted

def find_current_status_fallback(text: str) -> str | None:
    patterns = [r"pain is (minimal|mild)", r"occasional\s+([a-zA-Z\s]+)"]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return match.group(0) if match.lastindex is None else f"occasional {match.group(1).strip()}"
    return None

def build_summary_json(name: str | None, ner_data: dict, qa_data: dict) -> dict:
    return {
        "Patient_Name": name, "Symptoms": ner_data.get("Symptoms", []),
        "Diagnosis": qa_data.get("Diagnosis", None), "Treatment": list(set(ner_data.get("Treatment", []))),
        "Current_Status": qa_data.get("Current_Status", None), "Prognosis": qa_data.get("Prognosis", None),
        "Extracted_On": datetime.utcnow().isoformat() + "Z"
    }

def analyze_patient_sentiment(patient_dialogue: list) -> list:
    sentiment_classifier = get_sentiment_classifier()
    intent_classifier = get_intent_classifier()
    if not sentiment_classifier or not intent_classifier:
        st.error("Sentiment/Intent models not loaded. Cannot perform analysis.")
        return []
    
    results = []
    intent_labels = ["Seeking reassurance", "Reporting symptoms", "Expressing concern", "Expressing gratitude", "Reporting outcome"]
    for line in patient_dialogue:
        analysis = {"text": line}
        sentiment_result = sentiment_classifier(line)[0]
        analysis["sentiment"] = sentiment_result['label']
        analysis["sentiment_score"] = round(sentiment_result['score'], 2)
        intent_result = intent_classifier(line, candidate_labels=intent_labels)
        analysis["intent"] = intent_result['labels'][0]
        analysis["intent_score"] = round(intent_result['scores'][0], 2)
        results.append(analysis)
    return results

def generate_soap_note_hybrid(api_key, transcript_text):
    if not GENAI_AVAILABLE:
        return {"Error": "google.generativeai library not installed.", "Details": "Please run 'pip install google-generativeai'"}
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return {"Error": "Authentication failed. Please check your API Key.", "Details": str(e)}

    json_schema = """
    {"Subjective": {"Chief_Complaint": "Brief reason for visit.","History_of_Present_Illness": "Concise summary of event and symptoms."}, "Objective": {"Physical_Exam": "Key findings from physical exam.","Observations": "Short, objective observations."}, "Assessment": {"Diagnosis": "Primary diagnosis.","Severity": "One or two words on severity."}, "Plan": {"Treatment": "Concise treatment plan.","Follow_Up": "Brief follow-up instructions."}}
    """
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    prompt = f"You are an expert medical scribe. Convert the following transcript into a concise JSON object. Avoid narrative. Adhere strictly to this JSON schema: {json_schema}\n\nTranscript:\n---\n{transcript_text}\n---\nGenerate ONLY the JSON object."
    
    try:
        response = model.generate_content(prompt)
        cleaned_json_string = response.text.strip().replace("```json", "").replace("```", "")
        llm_output = json.loads(cleaned_json_string)
    except Exception as e:
        return {"Error": "Failed to generate valid JSON from the LLM.", "Details": str(e)}

    # Rule-based validation to ensure all keys are present
    final_soap_note = {"Subjective": {"Chief_Complaint": "Not specified","History_of_Present_Illness": "Not specified"},"Objective": {"Physical_Exam": "Not specified","Observations": "Not specified"},"Assessment": {"Diagnosis": "Not specified","Severity": "Not specified"},"Plan": {"Treatment": "Not specified","Follow_Up": "Not specified"}}
    for section, fields in final_soap_note.items():
        if section in llm_output and isinstance(llm_output.get(section), dict):
            for field, default_value in fields.items():
                final_soap_note[section][field] = llm_output[section].get(field, default_value)
    return final_soap_note


# --- STREAMLIT APP UI ---
st.set_page_config(page_title="AI Clinical Insights", layout="wide")
st.title("🩺 Physician Notetaker")
st.markdown("An all-in-one tool to extract structured data, analyze patient sentiment, and generate SOAP notes from clinical transcripts.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google AI Studio Key (for SOAP)", type="password")
    if not GENAI_AVAILABLE:
        st.warning("`google-generativeai` is not installed. SOAP Note feature is disabled. Install with `pip install google-generativeai`")
    st.divider()
    st.header("🤖 Model Status")
    if TRANSFORMERS_AVAILABLE:
        st.success("Transformers models loaded.")
    if GENAI_AVAILABLE and api_key:
        st.success("Google Generative AI configured.")

uploaded = st.file_uploader("Upload transcript (.txt)", type=["txt"])

if uploaded:
    transcript = read_txt(uploaded)
    
    st.subheader("1. Extracted Clinical Summary")
    with st.spinner("Running NER and QA models for summary..."):
        patient_name = extract_patient_name(transcript)
        medical_ner_results = extract_medical_info(transcript)
        qa_results = extract_qa_info(transcript)
        if not qa_results.get("Current_Status"):
            fallback_status = find_current_status_fallback(transcript)
            if fallback_status: qa_results["Current_Status"] = fallback_status
        summary_json = build_summary_json(patient_name, medical_ner_results, qa_results)
    st.json(summary_json)
    st.download_button("Download Summary JSON", data=json.dumps(summary_json, indent=2), file_name="patient_summary.json", mime="application/json")

    st.divider()

    st.subheader("2. Patient Sentiment & Intent Analysis")
    with st.expander("Click to view line-by-line patient sentiment analysis"):
        patient_dialogue = isolate_patient_dialogue(transcript)
        if not patient_dialogue:
            st.warning("No patient dialogue found. Ensure lines start with 'Patient:'.")
        else:
            with st.spinner("Analyzing patient dialogue for sentiment and intent..."):
                sentiment_results = analyze_patient_sentiment(patient_dialogue)
            df_results = pd.DataFrame(sentiment_results)
            st.dataframe(df_results, use_container_width=True)
            st.download_button("Download Sentiment JSON", data=json.dumps(sentiment_results, indent=2), file_name="patient_sentiment.json", mime="application/json")

    st.divider()

    st.subheader("3. Generate SOAP Note")
    if 'soap_note' not in st.session_state:
        st.session_state.soap_note = None
    
    if st.button("✨ Generate SOAP Note with Gemini AI", use_container_width=True):
        if not api_key:
            st.error("Please enter your Google AI Studio API Key in the sidebar to generate a SOAP note.")
        elif not GENAI_AVAILABLE:
            st.error("The 'google-generativeai' library is required for this feature.")
        else:
            with st.spinner("Generating SOAP note..."):
                soap_note_dict = generate_soap_note_hybrid(api_key, transcript)
                st.session_state.soap_note = json.dumps(soap_note_dict, indent=2)

    if st.session_state.soap_note:
        st.json(st.session_state.soap_note)
        st.download_button("Download SOAP Note JSON", data=st.session_state.soap_note, file_name="soap_note.json", mime="application/json")