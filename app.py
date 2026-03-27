import os
import json
import html
import warnings
import streamlit as st
import torch
import torch.nn as nn
import google.generativeai as genai
from transformers import AutoTokenizer, RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput
from safetensors.torch import load_file

warnings.filterwarnings("ignore")


# ----------------- MODEL DEFINITION -----------------
class RobertaBiLSTMCNNClassifier(nn.Module):
    def __init__(self, model_name, num_labels=7, lstm_hidden=256, lstm_layers=1,
                 bidirectional=True, num_filters=128, kernel_sizes=(2, 3, 4), dropout=0.35):
        super().__init__()
        self.num_labels = num_labels
        self.roberta = RobertaModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.roberta.config.hidden_size
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)
        self.convs = nn.ModuleList([nn.Conv1d(hidden_size, num_filters, k) for k in kernel_sizes])

        final_dim = hidden_size + lstm_out_dim + num_filters * len(kernel_sizes)
        self.fc_norm = nn.LayerNorm(final_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        roberta_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        sequence_output = roberta_outputs.last_hidden_state
        cls_embedding = sequence_output[:, 0, :]

        # BiLSTM path
        lstm_out, _ = self.lstm(sequence_output)
        lstm_out = self.lstm_norm(lstm_out)
        mask = attention_mask.unsqueeze(-1).float()
        lstm_pooled = (lstm_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # CNN path
        x_cnn = sequence_output.transpose(1, 2)
        cnn_pooled = torch.cat(
            [torch.max(torch.relu(conv(x_cnn)), dim=2).values for conv in self.convs], dim=1
        )

        combined = self.dropout(self.fc_norm(torch.cat([cls_embedding, lstm_pooled, cnn_pooled], dim=1)))
        return SequenceClassifierOutput(logits=self.classifier(combined))


# ----------------- CONFIG -----------------
MODEL_PATH = "./models/mental_health_roberta_hybrid_finale"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

LABEL_RISK = {
    "Normal": 0.1, "Stress": 0.4, "Anxiety": 0.5,
    "Depression": 0.7, "Personality disorder": 0.7,
    "Bipolar": 0.7, "Suicidal": 0.95,
}
VALID_LABELS = set(LABEL_RISK.keys())


# ----------------- LOAD CLASSIFIER -----------------
@st.cache_resource(show_spinner=True)
def load_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobertaBiLSTMCNNClassifier(MODEL_PATH)
    state_dict = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    return model, tokenizer, device

with open(f"{MODEL_PATH}/label_mapping.json") as f:
    label_mapping = json.load(f)

id_to_label = {v: k for k, v in label_mapping.items()}
model, tokenizer, device = load_classifier()


def risk_from_probs(probs):
    return float(sum(p * LABEL_RISK[id_to_label[i]] for i, p in enumerate(probs.tolist())))

def classify_one(text):
    inputs = tokenizer([text], return_tensors="pt", truncation=True,
                       max_length=512, padding=True).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=1).cpu()[0]
    pred_id = torch.argmax(probs).item()
    suicidal_idx = next((i for i, l in id_to_label.items() if l == "Suicidal"), None)
    return {
        "label": id_to_label[pred_id],
        "confidence": float(probs[pred_id]),
        "probs": probs,
        "risk": risk_from_probs(probs),
        "suicidal_prob": float(probs[suicidal_idx]) if suicidal_idx is not None else 0.0,
    }


# ----------------- SCORING -----------------
def compute_combined_risk(cls_risk, llm_label):
    return 0.65 * cls_risk + 0.35 * LABEL_RISK.get(llm_label, 0.1)

def compute_mental_score(risk_history):
    if not risk_history:
        return 0.0
    n = len(risk_history)
    weights = list(range(1, n + 1))
    weighted_avg = sum(w * r for w, r in zip(weights, risk_history)) / sum(weights)
    simple_mean = sum(risk_history) / n
    return round(0.65 * weighted_avg + 0.35 * simple_mean, 4)


# ----------------- GEMINI (single call) -----------------
def build_system_prompt(mode, mental_score, current_text):
    """
    Single call: Gemini returns a structured response with LABEL on line 1
    and the chat reply from line 2 onward. Verbosity is capped per mode.
    """
    labels_str = ", ".join(VALID_LABELS)

    verbosity = {
        "normal":    "Reply in 1-2 short casual sentences, like a friendly human. No lists, no long explanations.",
        "elevated":  "Reply in 2-3 sentences. Be warm and validating. Ask one gentle open question at most.",
        "high_risk": "Reply in 3-5 sentences. Be empathetic and grounded. Suggest one coping step. Gently recommend professional help.",
        "crisis":    "Be brief but clear. Immediately point to crisis resources. 2-4 sentences max. No clinical language.",
    }

    tone = {
        "normal":    "User seems stable. Friendly, light tone.",
        "elevated":  "User shows mild distress. Gentle and validating.",
        "high_risk": "User is significantly distressed. Careful and empathetic.",
        "crisis":    "CRISIS — prioritise immediate safety. Direct user to emergency help.",
    }

    return (
        f"You are a supportive mental health companion. You are NOT a therapist and cannot diagnose. This an Indian user.\n\n"
        f"Current risk score: {mental_score:.2f}. Mode: {mode}. {tone[mode]}\n\n"
        f"RESPONSE FORMAT — your reply must be exactly two parts:\n"
        f"Line 1: LABEL: <one of: {labels_str}> — classify ONLY this message in isolation: \"{current_text}\"\n"
        f"Line 2 onward: your actual reply to the user.\n\n"
        f"REPLY LENGTH: {verbosity[mode]}\n"
        f"Never reveal the label line to the user. Never mention scores or modes."
    )

def call_gemini(history, system_prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set GEMINI_API_KEY in environment.")
        st.stop()
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME, system_instruction=system_prompt)
    messages = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in history if m["role"] in ("user", "assistant", "model")
    ]
    try:
        return gemini.generate_content(messages).text.strip()
    except Exception as e:
        return f"LABEL: Normal\nSorry, I ran into an error: {e}"

def parse_gemini_response(raw):
    """Extract label from line 1 and reply from the rest."""
    lines = raw.strip().splitlines()
    llm_label = "Normal"
    reply_lines = lines

    if lines and lines[0].upper().startswith("LABEL:"):
        candidate = lines[0].split(":", 1)[1].strip()
        # fuzzy match in case Gemini adds punctuation
        for valid in VALID_LABELS:
            if valid.lower() in candidate.lower():
                llm_label = valid
                break
        reply_lines = lines[1:]

    reply = "\n".join(reply_lines).strip()
    return llm_label, reply


# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Mental Health Chatbot", page_icon="💬", layout="wide")

st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 0;
    overflow: hidden;
}
.chat-wrapper {
    height: 68vh;
    overflow-y: auto;
    padding: 14px 10px;
    border-radius: 12px;
    background-color: #111b21;
    margin-bottom: 6px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.chat-row-user      { display: flex; justify-content: flex-end; }
.chat-row-assistant { display: flex; justify-content: flex-start; }
.bubble-user {
    background-color: #005c4b;
    color: #e9edef;
    padding: 9px 14px;
    border-radius: 18px 18px 4px 18px;
    max-width: 68%;
    font-size: 0.93rem;
    line-height: 1.55;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.bubble-assistant {
    background-color: #202c33;
    color: #e9edef;
    padding: 9px 14px;
    border-radius: 18px 18px 18px 4px;
    max-width: 68%;
    font-size: 0.93rem;
    line-height: 1.55;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.sender-tag {
    font-size: 0.71rem;
    font-weight: 700;
    margin-bottom: 4px;
    opacity: 0.65;
    letter-spacing: 0.02em;
}
</style>
""", unsafe_allow_html=True)

st.title("💬 Mental Health Assistant")
st.caption(
    "⚠️ Not a substitute for professional care. "
    "If you are in crisis, contact emergency services or a crisis hotline immediately."
)

# Session state init
for key, default in [("history", []), ("risk_history", []), ("last_score", 0.0),
                     ("last_cls", None), ("last_llm_label", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("📊 Mental Health Score")
    score = st.session_state.last_score
    st.progress(min(max(score, 0.0), 1.0))
    st.markdown(f"**Score:** {score:.2f}")

    if st.session_state.last_cls:
        cls = st.session_state.last_cls
        st.divider()
        st.markdown("**🤖 Model Prediction**")
        st.write(f"Label: `{cls['label']}`")
        st.write(f"Confidence: {cls['confidence']:.2f}")
        st.write(f"Suicidal prob: {cls['suicidal_prob']:.2f}")

    if st.session_state.last_llm_label:
        st.markdown("**✨ LLM Quick Read**")
        st.write(f"Label: `{st.session_state.last_llm_label}`")
        st.write(f"Risk weight: {LABEL_RISK[st.session_state.last_llm_label]:.2f}")

    st.divider()
    if score > 0.7:
        st.warning("High risk detected. Encourage the user to seek professional help.")
    elif score > 0.4:
        st.info("Elevated distress detected. Use extra empathy and care.")
    else:
        st.success("Low to moderate distress. Continue supportive conversation.")

# ----------------- CHAT BUBBLES -----------------
chat_html = '<div class="chat-wrapper" id="chat-box">'
for msg in st.session_state.history:
    content = html.escape(msg["content"])
    if msg["role"] == "user":
        chat_html += f"""
        <div class="chat-row-user">
            <div class="bubble-user">
                <div class="sender-tag">You</div>{content}
            </div>
        </div>"""
    else:
        chat_html += f"""
        <div class="chat-row-assistant">
            <div class="bubble-assistant">
                <div class="sender-tag">Assistant 🤖</div>{content}
            </div>
        </div>"""

chat_html += """</div>
<script>
    const box = document.getElementById("chat-box");
    if (box) box.scrollTop = box.scrollHeight;
</script>"""

st.markdown(chat_html, unsafe_allow_html=True)

# ----------------- INPUT -----------------
user_input = st.chat_input("Type your message...")

if user_input:
    text = user_input.strip()
    st.session_state.history.append({"role": "user", "content": text})

    # ML classification (local, free)
    cls = classify_one(text)

    # Determine mode from current score before this message
    current_score = st.session_state.last_score
    if cls["suicidal_prob"] > 0.5:
        mode = "crisis"
    elif current_score > 0.7:
        mode = "high_risk"
    elif current_score > 0.4:
        mode = "elevated"
    else:
        mode = "normal"

    system_prompt = build_system_prompt(mode, current_score, text)

    # Single Gemini call — returns label + reply together
    raw = call_gemini(st.session_state.history, system_prompt)
    llm_label, reply = parse_gemini_response(raw)

    # Blend ML + LLM risk, update score
    combined_risk = compute_combined_risk(cls["risk"], llm_label)
    st.session_state.risk_history.append(combined_risk)
    st.session_state.last_score = compute_mental_score(st.session_state.risk_history)
    st.session_state.last_cls = cls
    st.session_state.last_llm_label = llm_label

    st.session_state.history.append({"role": "assistant", "content": reply})
    st.rerun()