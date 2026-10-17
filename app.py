import os

import httpx
import streamlit as st


API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="AI Document Insight", page_icon="\U0001f4ac")
st.title("Razgovor s agentom")


def api_call(method, route, **kwargs):
    try:
        response = httpx.request(method, API_URL + route, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get("detail", "Zahtjev nije uspio.")
        except ValueError:
            detail = "Zahtjev nije uspio."
        raise RuntimeError(detail if isinstance(detail, str) else "Provjeri unesene podatke.") from error
    except httpx.RequestError as error:
        raise RuntimeError("Backend nije dostupan ili je zahtjev istekao. Provjeri je li API pokrenut.") from error


if st.sidebar.button("Novi razgovor"):
    st.session_state.pop("api_session_id", None)
    st.session_state.pop("messages", None)
    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    st.rerun()

if "api_session_id" not in st.session_state:
    try:
        session = api_call("POST", "/sessions", json={"include_examples": True})
        st.session_state.api_session_id = session["session_id"]
        st.session_state.documents = session["documents"]
        st.session_state.messages = []
    except RuntimeError as error:
        st.error(str(error))
        if st.button("Poku\u0161aj ponovno"):
            st.rerun()
        st.stop()

session_id = st.session_state.api_session_id
with st.sidebar:
    st.subheader("Dokumenti")
    uploaded = st.file_uploader(
        "Odaberi PDF ili sliku", type=["pdf", "png", "jpg", "jpeg"],
        max_upload_size=30, key=f"upload-{st.session_state.get('upload_key', 0)}",
    )
    if st.button("Obradi dokument", disabled=uploaded is None):
        try:
            with st.spinner("Izdvajam tekst i pripremam dokument za razgovor\u2026"):
                result = api_call("POST", "/upload", data={"session_id": session_id},
                                  files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                st.session_state.documents = api_call(
                    "GET", "/documents", params={"session_id": session_id},
                )["documents"]
            if result["status"] == "unchanged":
                st.success("Dokument je ve\u0107 dostupan u razgovoru.")
            else:
                st.success("Dokument je spreman. Mo\u017ee\u0161 odmah pitati o njemu.")
            for warning in result["warnings"]:
                st.warning(warning)
        except RuntimeError as error:
            st.error(str(error))
    st.caption(f"Dokumenata u ovom razgovoru: {len(st.session_state.documents)}")
    if st.session_state.documents:
        with st.expander("Dostupni dokumenti"):
            for document in st.session_state.documents:
                st.text(document["filename"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Napi\u0161i poruku\u2026", submit_mode="disable", max_chars=2000)
if prompt and prompt.strip():
    prompt = prompt.strip()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent razmi\u0161lja\u2026"):
                result = api_call("POST", "/ask", json={"session_id": session_id, "message": prompt})
            st.markdown(result["answer"])
            st.session_state.messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": result["answer"]},
            ])
        except RuntimeError as error:
            st.error(str(error))
