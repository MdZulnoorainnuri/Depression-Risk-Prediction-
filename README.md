# 🧠 Depression Risk Prediction

An interactive web application built with **Streamlit** that calculates a screening score for depression risk based on self-reported symptoms. It optionally provides AI-powered explanations using the **Groq API (Llama-3.3-70b)** or falls back to a built-in rule-based explanation system.

---

## ✨ Features
- 📊 **Interactive Symptom Screening**: Rate sleep, mood, appetite, energy, concentration, and social withdrawal.
- ⚡ **Real-time Risk Scoring**: Instant score calculation from 0 to 100 with risk classification (Low, Moderate, High).
- 🤖 **AI Explanation (Groq LLM)**: Summarizes results with personalized guidance using `llama-3.3-70b-versatile`.
- 🛡️ **Fallback Local Mode**: Operates seamlessly offline or without an API key using built-in rule-based logic.
- 🔐 **Flexible Key Options**: Enter your GROQ API key in the sidebar, set it in `.env` / `apikey.txt`, or via Streamlit Secrets.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/MdZulnoorainnuri/Depression-Risk-Prediction-.git
cd Depression-Risk-Prediction-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run depressionriskprediction.py
```

---

## 🔑 Setting Up GROQ API Key (Optional)
To enable AI explanations, get a free API key from [Console Groq](https://console.groq.com):

- **Option A (Sidebar UI)**: Enter your key directly in the sidebar under `🔑 API Settings`.
- **Option B (File)**: Create an `apikey.txt` file in the project folder and paste your key.
- **Option C (Environment)**: Set `GROQ_API_KEY=your_key_here` in your environment.
- **Option D (Streamlit Cloud)**: Add `GROQ_API_KEY = "your_key_here"` under **Secrets**.

*If no key is provided, the app runs cleanly in local explanation mode.*

---

## ⚠️ Disclaimer
*This tool is for educational and screening purposes only. It does not provide medical diagnoses. If you are experiencing mental health difficulties, please consult a qualified healthcare professional.*
