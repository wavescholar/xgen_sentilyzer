import os
import streamlit as st
from openai import OpenAI

# # ------------- Debugging in VSCode
# import debugpy
# import streamlit as st
# import app

# # pylint: disable=invalid-name
# markdown = st.markdown(
#     """
# ## Ready to attach the VS Code Debugger
# for more info (https://awesome-streamlit.readthedocs.io/en/latest/vscode.html#integrated-debugging)
# """
# )

# if not debugpy.is_client_connected():
#     debugpy.listen(5679)
#     debugpy.wait_for_client()

# # -------------------------------------------------

st.markdown("# OpenAI Model")
st.sidebar.markdown("#  page ")

API_KEY = os.environ.get("OPENAI_APIKEY")

client = OpenAI(api_key=API_KEY)

# with st.sidebar:
# openai_api_key = st.text_input(
#     "OpenAI API Key",
#     key="chatbot_api_key",
#     type="password",
# )

st.title(" Chatbot  ")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt_openai := st.chat_input("main_page"):
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt_openai})
    st.chat_message("user").write(prompt_openai)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=st.session_state.messages
    )
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)
