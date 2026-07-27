import os
from pathlib import Path

import requests
import streamlit as st

API_KEY_FILE = Path(__file__).with_name("apikey.txt")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT_SECONDS = 30


def load_raw_api_key() -> str | None:
    """Return the raw API key from the environment or apikey.txt file."""
    env_key = os.environ.get("GROQ_API_KEY")
    if env_key is not None:
        return env_key
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text(encoding="utf-8")
    return None


def normalize_api_key(raw_key: str | None) -> str:
    """Normalize the raw API key and return an empty string for invalid values."""
    if raw_key is None:
        return ""
    return raw_key.strip()


def calculate_risk_score(
    sleep_hours: float,
    mood: int,
    appetite: int,
    energy: int,
    concentration: int,
    social_activity: int,
) -> int:
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


def classify_risk(score: int) -> str:
    if score >= 75:
        return "High risk"
    if score >= 45:
        return "Moderate risk"
    return "Low risk"


def build_groq_prompt(inputs: dict, score: int, classification: str) -> str:
    lines = [
        "You are a compassionate mental health assistant.",
        "The user has provided the following symptom ratings:",
    ]
    for key, value in inputs.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            f"Calculated risk score: {score} out of 100",
            f"Risk classification: {classification}",
            "Please provide a short explanation of what this means and suggest healthy next steps, without giving medical advice.",
        ]
    )
    return "\n".join(lines)


def generate_local_explanation(inputs: dict, score: int, classification: str) -> str:
    explanation_lines = [
        f"Your estimated risk score is {score} out of 100, which is classified as {classification}.",
        "This score is a simple screening estimate based on self-reported symptoms, not a diagnosis.",
        "Key observations:",
    ]

    for key, value in inputs.items():
        explanation_lines.append(f"- {key}: {value}")

    if classification == "High risk":
        explanation_lines.append(
            "A high risk score may indicate a stronger need for professional support, especially if these symptoms have worsened or lasted more than two weeks."
        )
    elif classification == "Moderate risk":
        explanation_lines.append(
            "A moderate risk score suggests some concerning symptoms that may benefit from closer monitoring and self-care."
        )
    else:
        explanation_lines.append(
            "A low risk score means your current symptoms are milder, but it is still important to stay aware of changes over time."
        )

    explanation_lines.extend(
        [
            "Suggested healthy next steps:",
            "- Keep a regular sleep schedule and aim for consistent sleep duration.",
            "- Stay connected with supportive friends or family.",
            "- Engage in physical activity, even short daily walks.",
            "- Practice stress-reduction techniques like breathing exercises, journaling, or mindfulness.",
            "- Reach out to a qualified healthcare provider if symptoms persist or worsen.",
        ]
    )
    return "\n".join(explanation_lines)


def build_groq_request_payload(prompt: str) -> dict:
    return {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }


def parse_groq_response(response_json: dict) -> str:
    if not isinstance(response_json, dict):
        raise ValueError("Unexpected response format from Groq API.")

    try:
        return response_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected Groq response structure.") from exc


def query_groq_api(api_key: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = build_groq_request_payload(prompt)

    response = requests.post(
        GROQ_API_URL,
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    if response.status_code == 401:
        raise requests.HTTPError("Invalid GROQ API Key. Please check apikey.txt.", response=response)
    if response.status_code == 404:
        raise requests.HTTPError("The selected model does not exist.", response=response)
    if response.status_code == 429:
        raise requests.HTTPError("Rate limit exceeded. Please wait and try again.", response=response)
    if response.status_code == 500:
        raise requests.HTTPError("Groq server error. Please try again later.", response=response)

    response.raise_for_status()
    return parse_groq_response(response.json())


def is_network_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in [
            "failed to resolve",
            "name resolution",
            "getaddrinfo failed",
            "max retries exceeded",
            "temporary failure in name resolution",
            "could not connect",
            "connection aborted",
            "connection reset",
        ]
    )


def display_local_fallback(inputs: dict, score: int, classification: str) -> None:
    local_explanation = generate_local_explanation(inputs, score, classification)
    st.text_area("Local fallback explanation", local_explanation, height=260)


def main() -> None:
    st.set_page_config(
        page_title="Depression Risk Prediction",
        page_icon="🧠",
        layout="centered",
    )

    st.title("Depression Risk Prediction")
    st.markdown(
        "This app calculates a simple depression-risk score from self-reported symptoms and can optionally summarize the result using a GROQ model."
    )

    raw_api_key = load_raw_api_key()
    api_key = normalize_api_key(raw_api_key)

    if raw_api_key is None:
        st.warning(
            "No GROQ API key found. Create an `apikey.txt` file or set the `GROQ_API_KEY` environment variable to enable the model explanation feature."
        )
    elif api_key == "":
        st.warning("Invalid API Key.")

    with st.form("risk_form"):
        st.subheader("Symptom input")
        sleep_hours = st.slider("Average sleep hours per night", 0.0, 12.0, 6.5, 0.5)
        mood = st.slider("Mood level (1 = good, 5 = very low)", 1, 5, 3)
        appetite = st.slider("Appetite changes (1 = normal, 5 = very poor)", 1, 5, 3)
        energy = st.slider("Energy level (1 = normal, 5 = very low)", 1, 5, 3)
        concentration = st.slider("Concentration difficulties (1 = none, 5 = severe)", 1, 5, 3)
        social_activity = st.slider(
            "Interest in social activities (1 = normal, 5 = withdrawn)", 1, 5, 3
        )
        submitted = st.form_submit_button("Predict risk")

    if submitted:
        inputs = {
            "Sleep hours": f"{sleep_hours:.1f}",
            "Mood level": mood,
            "Appetite changes": appetite,
            "Energy level": energy,
            "Concentration difficulties": concentration,
            "Social withdrawal": social_activity,
        }
        score = calculate_risk_score(
            sleep_hours, mood, appetite, energy, concentration, social_activity
        )
        classification = classify_risk(score)

        st.metric(label="Depression risk score", value=f"{score}/100", delta=classification)
        st.write(
            "### Interpretation",
            f"**{classification}** — This is a simple screening estimate. It is not a diagnosis."
        )

        if api_key:
            st.write("### GROQ model explanation")
            prompt = build_groq_prompt(inputs, score, classification)
            try:
                explanation = query_groq_api(api_key, prompt)
                st.text_area("GROQ explanation", explanation, height=220)
            except requests.exceptions.Timeout:
                st.error("Connection timed out while contacting Groq.")
                st.info(
                    "Falling back to a local explanation. You can still use the risk score shown above. "
                    "The GROQ explanation feature requires outbound HTTPS access to api.groq.com."
                )
                display_local_fallback(inputs, score, classification)
            except requests.HTTPError as error:
                message = str(error)
                if "Invalid GROQ API Key" in message:
                    st.error("Invalid GROQ API Key. Please check apikey.txt.")
                elif "The selected model does not exist" in message:
                    st.error("The selected model does not exist.")
                elif "Rate limit exceeded" in message:
                    st.error("Rate limit exceeded. Please wait and try again.")
                elif "Groq server error" in message:
                    st.error("Groq server error. Please try again later.")
                else:
                    st.error(f"Failed to call Groq API: {message}")
                st.info(
                    "Falling back to a local explanation. You can still use the risk score shown above. "
                    "The GROQ explanation feature requires outbound HTTPS access to api.groq.com."
                )
                display_local_fallback(inputs, score, classification)
            except requests.exceptions.RequestException as error:
                if is_network_error(error):
                    st.error(
                        "Unable to connect to the Groq API. Check your internet connection, DNS settings, firewall, proxy, or VPN."
                    )
                else:
                    st.error(f"Failed to call Groq API: {error}")
                st.info(
                    "Falling back to a local explanation. You can still use the risk score shown above. "
                    "The GROQ explanation feature requires outbound HTTPS access to api.groq.com."
                )
                display_local_fallback(inputs, score, classification)
            except ValueError as error:
                st.error(f"Failed to parse Groq response: {error}")
                st.info(
                    "Falling back to a local explanation. You can still use the risk score shown above. "
                    "The GROQ explanation feature requires outbound HTTPS access to api.groq.com."
                )
                display_local_fallback(inputs, score, classification)
        else:
            st.write("### Local explanation")
            st.text_area("Explanation", generate_local_explanation(inputs, score, classification), height=260)

    st.markdown("---")
    st.info(
        "This tool is for educational use only. If you are concerned about your mental health, please contact a qualified professional."
    )


if __name__ == "__main__":
    main()
