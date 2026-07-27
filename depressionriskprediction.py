"""
Depression Risk Prediction Application
Calculates depression screening scores and provides AI (Groq) or local rule-based explanations.
"""

import os
from pathlib import Path

import requests
import streamlit as st

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 30


def load_api_key(sidebar_input: str = "") -> str:
    """Load API key in priority: 1) Sidebar input, 2) st.secrets, 3) Environment variable, 4) apikey.txt file."""
    if sidebar_input and sidebar_input.strip():
        return sidebar_input.strip()

    try:
        if "GROQ_API_KEY" in st.secrets and str(st.secrets["GROQ_API_KEY"]).strip():
            return str(st.secrets["GROQ_API_KEY"]).strip()
        if "groq_api_key" in st.secrets and str(st.secrets["groq_api_key"]).strip():
            return str(st.secrets["groq_api_key"]).strip()
    except Exception:
        pass

    env_key = os.environ.get("GROQ_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    api_file = Path(__file__).with_name("apikey.txt")
    if api_file.exists():
        try:
            key = api_file.read_text(encoding="utf-8").strip()
            if key and not key.startswith("#"):
                return key
        except Exception:
            pass

    return ""


def calculate_risk_score(
    sleep_hours: float,
    mood: int,
    appetite: int,
    energy: int,
    concentration: int,
    social_activity: int,
) -> int:
    """Calculate depression risk score (0-100) based on symptom inputs."""
    sleep_penalty = 0
    if sleep_hours <= 4:
        sleep_penalty = 30
    elif sleep_hours <= 6:
        sleep_penalty = 20
    elif sleep_hours <= 8:
        sleep_penalty = 5
    else:
        sleep_penalty = 15

    mood_penalty = (mood - 1) * 15
    appetite_penalty = (appetite - 1) * 12
    energy_penalty = (energy - 1) * 13
    concentration_penalty = (concentration - 1) * 12
    social_penalty = (social_activity - 1) * 12

    raw_score = (
        sleep_penalty
        + mood_penalty
        + appetite_penalty
        + energy_penalty
        + concentration_penalty
        + social_penalty
    )
    return min(max(int(raw_score), 0), 100)


def classify_risk(score: int) -> tuple[str, str, str]:
    """Return risk classification tuple: (label, badge_color_type, emoji)."""
    if score >= 75:
        return ("High Risk", "red", "🔴")
    if score >= 45:
        return ("Moderate Risk", "orange", "🟠")
    return ("Low Risk", "green", "🟢")


def get_groq_explanation(inputs: dict, score: int, classification: str, api_key: str) -> tuple[str | None, str | None]:
    """
    Call Groq API to get an AI-generated explanation.
    Returns (explanation_markdown, error_message).
    """
    system_prompt = (
        "You are a supportive mental health education assistant. "
        "You never diagnose. "
        "Explain results in simple language. "
        "Use bullet points. "
        "Suggest healthy coping strategies. "
        "End with a reminder to seek professional help if symptoms persist."
    )

    user_content = [
        "The user provided the following symptom assessment ratings:",
    ]
    for key, value in inputs.items():
        user_content.append(f"- {key}: {value}")

    user_content.extend([
        f"Calculated Risk Score: {score}/100",
        f"Risk Level: {classification}",
        "Please provide a structured summary with observations and practical healthy next steps."
    ])

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(user_content)},
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        
        if response.status_code == 401:
            return None, "Invalid API key."
        if response.status_code == 404:
            return None, "Model unavailable."
        if response.status_code == 429:
            return None, "Rate limit exceeded."
        if response.status_code >= 500:
            return None, "Groq server error."

        response.raise_for_status()
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return content, None

    except requests.exceptions.Timeout:
        return None, "Groq request timed out."
    except requests.exceptions.RequestException:
        return None, "Cannot connect to Groq."
    except (KeyError, IndexError, ValueError, TypeError):
        return None, "Unexpected response format from Groq."


def generate_local_explanation(inputs: dict, score: int, classification: str) -> str:
    """Generate structured markdown local explanation matching the AI format."""
    lines = [
        "## Summary",
        f"Based on your self-reported symptoms, your estimated depression risk score is **{score}/100** ({classification}).",
        "",
        "### Explanation & Observations",
    ]

    for key, value in inputs.items():
        lines.append(f"- **{key}**: {value}")

    lines.append("")
    if classification == "High Risk":
        lines.append(
            "Your symptom ratings indicate elevated intensity across key areas. "
            "This suggests a stronger potential need for supportive care and professional consultation."
        )
    elif classification == "Moderate Risk":
        lines.append(
            "Your symptom ratings indicate moderate intensity. "
            "Close self-monitoring, stress reduction, and healthy daily habits are recommended."
        )
    else:
        lines.append(
            "Your symptom ratings indicate mild or low intensity. "
            "Continuing positive health habits and maintaining active social connections will support ongoing well-being."
        )

    lines.extend([
        "",
        "### Suggested Healthy Next Steps",
        "- **Consistent Sleep Routine**: Aim for 7–9 hours of regular, restful sleep every night.",
        "- **Physical Activity**: Engage in daily light movement, such as a 20-minute outdoor walk.",
        "- **Social Connection**: Stay in regular touch with supportive friends, family, or community.",
        "- **Mindfulness & Stress Reduction**: Practice daily deep breathing exercises, journaling, or meditation.",
        "- **Professional Consultation**: Speak with a licensed medical professional if symptoms persist or interfere with daily life.",
    ])

    return "\n".join(lines)


def display_results(inputs: dict, score: int, classification: str, color_type: str, emoji: str, api_key: str) -> None:
    """Display assessment results using structured Streamlit metrics, cards, and expanders."""
    st.markdown("## 📊 Assessment Results")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Risk Score", value=f"{score}/100")
    with col2:
        st.markdown(f"### Risk Level: {emoji} **{classification}**")
        if color_type == "red":
            st.error("High risk detected. Professional consultation is strongly advised.")
        elif color_type == "orange":
            st.warning("Moderate risk detected. Close self-monitoring is recommended.")
        else:
            st.success("Low risk detected. Continue maintaining healthy habits.")

    st.markdown("---")

    explanation_content = None
    used_ai = False

    if api_key:
        with st.spinner("Generating AI-powered explanation..."):
            explanation_content, err_msg = get_groq_explanation(inputs, score, classification, api_key)
            if explanation_content:
                used_ai = True
            else:
                st.info(f"ℹ️ {err_msg} Falling back to local explanation.")

    if not used_ai:
        explanation_content = generate_local_explanation(inputs, score, classification)

    expander_title = "🤖 AI-Powered Explanation" if used_ai else "📋 Structured Local Explanation"
    with st.expander(expander_title, expanded=True):
        st.markdown(explanation_content)


def main() -> None:
    st.set_page_config(
        page_title="Depression Risk Prediction",
        page_icon="🧠",
        layout="centered",
    )

    # Sidebar: AI Settings
    st.sidebar.markdown("### 🔑 AI Settings")
    st.sidebar.markdown("Enter your Groq API key below to enable AI-powered explanations:")
    
    user_api_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key from https://console.groq.com",
        placeholder="gsk_...",
    )

    api_key = load_api_key(user_api_key)

    if api_key:
        st.sidebar.success("✅ AI explanations enabled")
    else:
        st.sidebar.info("ℹ AI explanations are optional. The app is running with the built-in local explanation.")

    # Main Header
    st.title("Depression Risk Prediction")
    st.markdown(
        "This tool calculates a depression-risk score based on self-reported symptoms and provides an explanation using Groq LLM or a built-in local engine."
    )

    # Form
    with st.form("risk_form"):
        st.subheader("Symptom Input")
        sleep_hours = st.slider("Average sleep hours per night", 0.0, 12.0, 6.5, 0.5)
        mood = st.slider("Mood level (1 = good, 5 = very low)", 1, 5, 3)
        appetite = st.slider("Appetite changes (1 = normal, 5 = very poor)", 1, 5, 3)
        energy = st.slider("Energy level (1 = normal, 5 = very low)", 1, 5, 3)
        concentration = st.slider("Concentration difficulties (1 = none, 5 = severe)", 1, 5, 3)
        social_activity = st.slider(
            "Interest in social activities (1 = normal, 5 = withdrawn)", 1, 5, 3
        )
        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        inputs = {
            "Sleep Hours": f"{sleep_hours:.1f} hrs",
            "Mood Level": f"{mood}/5",
            "Appetite Changes": f"{appetite}/5",
            "Energy Level": f"{energy}/5",
            "Concentration Difficulties": f"{concentration}/5",
            "Social Withdrawal": f"{social_activity}/5",
        }
        score = calculate_risk_score(
            sleep_hours, mood, appetite, energy, concentration, social_activity
        )
        classification, color_type, emoji = classify_risk(score)

        display_results(inputs, score, classification, color_type, emoji, api_key)

    # Single Global Disclaimer (Requirement 10)
    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer**: This tool is for educational and screening purposes only. "
        "It does not provide a medical diagnosis. If you are concerned about your mental health, please consult a qualified healthcare professional."
    )


if __name__ == "__main__":
    main()
