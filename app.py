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


try:
    conversations = api_call("GET", "/sessions")
    if "session" not in st.session_state:
        st.session_state.session = (api_call("GET", f"/sessions/{conversations[0]['session_id']}") if conversations
                                    else api_call("POST", "/sessions", json={"include_examples": True}))
    session_id = st.session_state.session["session_id"]
    st.session_state.session = api_call("GET", f"/sessions/{session_id}")
    titles = {item["session_id"]: item["title"] for item in conversations}
    empty = session_id not in titles and st.session_state.session["messages_used"] == 0
    if st.sidebar.button("Novi razgovor", disabled=empty) and not empty:
        st.session_state.session = api_call("POST", "/sessions", json={"include_examples": True})
        st.session_state.session["messages"] = []
        session_id = st.session_state.session["session_id"]
        st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    if session_id not in titles:
        titles = {session_id: "Novi razgovor", **titles}
    # Reset the widget when names/order change; a stale option index must not switch chats.
    choices = list(titles.items())
    if choices != st.session_state.get("conversation_choices"):
        st.session_state.conversation_choices = choices
        st.session_state.conversation_version = st.session_state.get("conversation_version", 0) + 1
    selected = st.sidebar.selectbox("Razgovori", list(titles), index=list(titles).index(session_id),
                                    format_func=titles.get, key=f"conversation-{st.session_state.conversation_version}")
    if selected != session_id:
        st.session_state.session = api_call("GET", f"/sessions/{selected}")
        st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1
    st.sidebar.caption("Razgovori se spremaju lokalno. Vidljivi su svima koji koriste ovu aplikaciju.")
except RuntimeError as error:
    st.error(str(error))
    if st.button("Pokušaj ponovno"):
        st.rerun()
    st.stop()

session = st.session_state.session
session_id = session["session_id"]


def refresh_documents():
    session.update(api_call("GET", f"/sessions/{session_id}"))


with st.sidebar:
    st.subheader("Dokumenti")
    uploaded = st.file_uploader("Odaberi PDF ili sliku", type=["pdf", "png", "jpg", "jpeg"],
                               max_upload_size=30, key=f"upload-{st.session_state.get('upload_key', 0)}")
    if st.button("Obradi dokument", disabled=uploaded is None):
        try:
            with st.spinner("Izdvajam tekst i pripremam dokument za razgovor…"):
                result = api_call("POST", "/upload", data={"session_id": session_id},
                                  files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
            st.session_state.upload_result = result
            st.rerun()
        except RuntimeError as error:
            st.error(str(error))
    if result := st.session_state.pop("upload_result", None):
        st.success("Dokument je već dostupan u razgovoru." if result["status"] == "unchanged"
                   else "Dokument je spreman. Možeš odmah pitati o njemu.")
        for warning in result["warnings"]:
            st.warning(warning)
    st.caption(f"Dokumenata u ovom razgovoru: {len(session['documents'])}")
    st.caption(f"Poruke u sesiji: {session['messages_used']}/{session['message_limit']}")
    if session["documents"]:
        with st.expander("Dostupni dokumenti"):
            for document in session["documents"]:
                st.text(document["filename"])

for message in session["messages"]:
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
                api_call("POST", "/ask", json={"session_id": session_id, "message": prompt})
            st.rerun()
        except RuntimeError as error:
            st.error(str(error))
            try:
                refresh_documents()
                if session["notice"]:
                    st.info(session["notice"])
            except RuntimeError:
                pass
