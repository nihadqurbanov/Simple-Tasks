import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from agent.agent import build_agent

load_dotenv()

st.set_page_config(page_title="AI Chatbot Agent", page_icon="🤖", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def get_agent(provider: str, model: str):
    agent_key = f"{provider}:{model}"
    if st.session_state.get("agent_key") != agent_key:
        st.session_state.agent = build_agent(provider, model)
        st.session_state.agent_key = agent_key
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
    return st.session_state.agent


with st.sidebar:
    st.header("Settings")
    provider = st.selectbox("Provider", ["Ollama", "OpenAI"])
    model_options = (
        ["gpt-oss:120b-cloud"]
        if provider == "Ollama"
        else ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )
    model = st.selectbox("Model", model_options)
    if provider == "Ollama":
        st.caption("Ollama Cloud — lokal qurasdirma teleb olunmur.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

st.title("AI Chatbot Agent")
st.caption("LangChain agent with calculator and datetime tools")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Mesajinizi yazin..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Dusunur..."):
            try:
                agent = get_agent(provider, model)
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                response = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                )
                ai_messages = [
                    message
                    for message in response["messages"]
                    if isinstance(message, AIMessage) and message.content
                ]
                answer = ai_messages[-1].content if ai_messages else "Cavab alina bilmedi."
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:
                error_text = f"Xeta: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text}
                )
