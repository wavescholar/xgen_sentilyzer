import os
import streamlit as st
import vertexai
from vertexai.preview.generative_models import GenerativeModel, ChatSession

project_id = "totemic-tower-407319"  # "wavelang"
location = "us-central1"
vertexai.init(project=project_id, location=location)

model = GenerativeModel("gemini-pro")
chat = model.start_chat()

st.markdown("# Google Model")
st.sidebar.markdown("#  page 2")

API_KEY = os.environ.get("GCP_GEMINI_APIKEY")


def get_chat_response(chat: ChatSession, prompt: str) -> str:
    response = chat.send_message(prompt)
    return response.text


# with st.sidebar:
# openai_api_key = st.text_input(
#     "OpenAI API Key",
#     key="chatbot_api_key",
#     type="password",
# )

st.title(" Chatbot Gemini ")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_gemini := st.chat_input("page_2"):
    st.session_state.messages.append({"role": "user", "content": prompt_gemini})
    st.chat_message("user").write(prompt_gemini)

    response = get_chat_response(chat, prompt_gemini)

    msg = response

    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg)
