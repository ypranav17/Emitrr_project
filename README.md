# Emitrr_project
🩺 Physician Notetaker
This project is a comprehensive tool designed to process unstructured clinical transcripts and convert them into valuable, structured data. By leveraging a suite of modern Natural Language Processing (NLP) models, this application automates the tedious process of manual data extraction, saving time for healthcare professionals and reducing the risk of error.

The application, built with Streamlit, provides three core functionalities:

Structured Clinical Summary: Extracts key medical information (symptoms, diagnosis, treatment, etc.) into a clean JSON format.

Patient Sentiment Analysis: Performs a line-by-line analysis of the patient's dialogue to identify sentiment (Anxious, Neutral, Reassured) and intent.

Generative SOAP Note: Utilizes a powerful Large Language Model (LLM) to generate a complete, well-structured SOAP note from the transcript.
(I ran out of time to generate a dataset for this task and train the NLP model as there is no solid data set available.)

## 🛠 Tech Stack

- **Application Framework:** Streamlit  

- **NLP & AI:**  
  - Hugging Face `transformers` for local NLP models  
  - `PyTorch` as the backend for transformer models  
  - Google `generativeai` for the generative SOAP note feature  

- **Core Libraries:** Pandas, Scikit-learn (for model metrics)  

- **Models Used:**  
  - **Medical NER:** `d4data/biomedical-ner-all`  
  - **General NER (Names):** `dbmdz/bert-large-cased-finetuned-conll03-english`  
  - **Question Answering:** `deepset/roberta-large-squad2`  
  - **Custom Sentiment:** Fine-tuned `distilbert-base-uncased`  
  - **Intent Detection:** `facebook/bart-large-mnli` (Zero-Shot)  
  - **SOAP Generation:** Google `Gemini-2.5-PRO`

## ⚙️ Setup and Installation

Follow these instructions to set up and run the project locally.

---

### 1. Prerequisites
- Python **3.11** (recommended to avoid dependency issues).  
- A **Google AI Studio API Key** for the SOAP Note generation feature.  

---

### 2. Clone the Repository
```bash
git clone https://github.com/ypranav17/Emitrr_project.git
cd Emitrr_project
```
### 3. Create and Activate Virtual Environment
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install google-generativeai
```


