from streamlit_feedback import streamlit_feedback
import streamlit as st
from dotenv import load_dotenv, find_dotenv
import openai
import os
import re

load_dotenv(find_dotenv(), override=True)

# openai.api_key = os.environ["OPENAI_API_KEY"]
ticker = st.session_state["ticker"]
year = st.session_state["year"]
user_proxy = st.session_state['user_proxy']
tool_proxy = st.session_state['tool_proxy'] 
# financial_docs_crew = st.session_state["finance_docs_crew"]
st.title(f"{ticker}-{year}")


def generate_response(input_text):
    # res = financial_docs_crew.kickoff({"question": input_text})
    chat_result = user_proxy.initiate_chat(
        recipient=tool_proxy,
        message=input_text,
        max_turns=10
    )
    res = ""
    for ct in chat_result.chat_history:
        if ct['role'] == 'assistant':
            # print(ct['content'])
            if ct['content'] is not None:
                res+=ct['content']+"\n\n"
    return res


if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, how can I help you?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Answering..."):
            docs = generate_response(prompt)
            docs = re.sub(r"\$", r"\\$", docs)
            docs = re.sub("TERMINATE", "", docs)
            st.write(docs)
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="Please describe the feedback in detail",
            )
    message = {"role": "assistant", "content": docs}
    st.session_state.messages.append(message)
