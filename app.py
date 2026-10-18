"""Streamlit chat; all document processing and agent calls go through the API."""

import httpx
import streamlit as st

from app_logging import log_event
from config import API_URL

st.set_page_config(page_title="AI Document Insight", page_icon="💬")
st.title("Razgovor s agentom")


def api_call(method, route, **kwargs):
    try:
        response = httpx.request(method, API_URL + route, timeout=300, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as error:
        log_event("ui_http_error", level="WARNING", interface="streamlit", path=route,  #logging
                  status_code=error.response.status_code, request_id=error.response.headers.get("X-Request-ID"))
        try:
            detail = error.response.json().get("detail")
        except ValueError:
            detail = None
        raise RuntimeError(detail if isinstance(detail, str) else "Zahtjev nije uspio.") from error
    except httpx.RequestError as error:
        log_event("ui_connection_error", level="ERROR", interface="streamlit", path=route, error_type=type(error).__name__)  #logging
        raise RuntimeError("Backend nije dostupan ili je zahtjev istekao. Provjeri je li API pokrenut.") from error


if st.sidebar.button("Novi razgovor"):
    st.session_state.pop("session", None)
    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    st.rerun()

if "session" not in st.session_state:
    try:
        st.session_state.session = api_call("POST", "/sessions", json={"include_examples": True})
        st.session_state.messages = []
    except RuntimeError as error:
        st.error(str(error))
        if st.button("Pokušaj ponovno"):
            st.rerun()
        st.stop()

session = st.session_state.session
session_id = session["session_id"]


def refresh_documents():
    session.update(api_call("GET", "/documents", params={"session_id": session_id}))


with st.sidebar:
    st.subheader("Dokumenti")
    uploaded = st.file_uploader("Odaberi PDF ili sliku", type=["pdf", "png", "jpg", "jpeg"],
                               max_upload_size=30, key=f"upload-{st.session_state.get('upload_key', 0)}")
    if st.button("Obradi dokument", disabled=uploaded is None):
        try:
            with st.spinner("Izdvajam tekst i pripremam dokument za razgovor…"):
                result = api_call("POST", "/upload", data={"session_id": session_id},
                                  files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                refresh_documents()
            st.success("Dokument je već dostupan u razgovoru." if result["status"] == "unchanged"
                       else "Dokument je spreman. Možeš odmah pitati o njemu.")
            for warning in result["warnings"]:
                st.warning(warning)
        except RuntimeError as error:
            st.error(str(error))
    st.caption(f"Dokumenata u ovom razgovoru: {len(session['documents'])}")
    st.caption(f"Poruke u sesiji: {session['messages_used']}/{session['message_limit']}")
    if session["documents"]:
        with st.expander("Dostupni dokumenti"):
            for document in session["documents"]:
                st.text(document["filename"])

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if session["notice"]:
    st.info(session["notice"])
prompt = st.chat_input("Napiši poruku…", submit_mode="disable", max_chars=2000,
                       disabled=session["messages_remaining"] == 0)
if prompt and prompt.strip():
    prompt = prompt.strip()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent razmišlja…"):
                result = api_call("POST", "/ask", json={"session_id": session_id, "message": prompt})
            session.update(result)
            st.session_state.messages.extend([
                {"role": "user", "content": prompt}, {"role": "assistant", "content": result["answer"]}])
            st.rerun()
        except RuntimeError as error:
            st.error(str(error))
            try:
                refresh_documents()
                if session["notice"]:
                    st.info(session["notice"])
            except RuntimeError:
                pass
