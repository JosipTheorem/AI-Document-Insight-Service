import streamlit as st

from agent import build_agent


st.set_page_config(page_title="AI Document Insight", page_icon="💬")
st.title("Razgovor s agentom")

if "document_agent" not in st.session_state:
    st.session_state.document_agent = build_agent()
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Napiši poruku…", submit_mode="disable")
if prompt and prompt.strip():
    prompt = prompt.strip()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent razmišlja…"):
                result = st.session_state.document_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    config={"configurable": {"thread_id": "streamlit-chat"}},
                )
                answer = result["messages"][-1].text
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as error:
            st.error(f"Odgovor nije uspio. Provjeri je li Ollama pokrenuta. {error}")
