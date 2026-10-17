"""Run with: python -m streamlit run monitoring/dashboard.py --server.port 8502"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from app_logging import log_path
from monitoring.data import filter_days, read_events, summarize, token_buckets

st.set_page_config(page_title="Monitoring aplikacije", page_icon="📊", layout="wide")
st.title("Monitoring aplikacije")
st.caption("Čita stvarne događaje iz tekstualnog loga. Automatsko osvježavanje svakih 5 sekundi.")
today = datetime.now(ZoneInfo("Europe/Zagreb")).date()
dates = st.sidebar.date_input("Razdoblje", value=(today, today), format="DD.MM.YYYY")
st.sidebar.caption("Datumi i grafovi: Europe/Zagreb")
st.sidebar.caption(f"Log: {log_path()}")
st.sidebar.button("Osvježi")


@st.fragment(run_every="5s")
def render():
    events, skipped = read_events(log_path())
    if len(dates) != 2:
        st.info("Odaberi početni i završni datum.")
        return
    events = filter_days(events, *dates)
    if skipped:
        st.caption(f"Preskočeno nepotpunih/neispravnih zapisa: {skipped}. Aktivni zapis može biti dovršen pri sljedećem čitanju.")
    if events.empty:
        st.info("Nema zapisa u odabranom razdoblju. Pošalji poruku ili uploadaj dokument.")
        return
    metrics = summarize(events)
    top = st.columns(4)
    for column, label, value in zip(top, ["Ulazni tokeni", "Izlazni tokeni", "Obrađene poruke", "Zahtjevi s internom greškom"],
                                    [metrics["input_tokens"], metrics["output_tokens"], metrics["answers"], metrics["errors"]]):
        column.metric(label, f"{value:,}")
    timing = st.columns(4)
    for column, label, key in zip(timing[:3], ["Medijan odgovora", "p95", "p99"], ["median", "p95", "p99"]):
        value = metrics[key]
        column.metric(label, "—" if pd.isna(value) else f"{value:.2f} s")
    timing[3].metric("Odbijeni HTTP zahtjevi (4xx)", metrics["rejected"])
    st.caption(f"Trajanje: cijeli uspješan /ask zahtjev, uključujući čekanje, security, pretragu i generiranje; "
               f"uključuje i terminal ako je korišten. Uzorak: {metrics['answers']} odgovora. "
               f"Sigurnosna odbijanja: {metrics['blocked']}. Ukupno HTTP zahtjeva: {metrics['requests']}.")
    if metrics["missing_usage"]:
        st.warning(f"Za {metrics['missing_usage']} završenih LLM poziva nema token metapodataka. Nisu procijenjeni niti pribrojeni.")

    st.subheader("Tokeni kroz vrijeme")
    minute, second = st.tabs(["Po minuti", "Po sekundi"])
    for container, frequency in ((minute, "min"), (second, "s")):
        with container:
            buckets = token_buckets(events, frequency)
            if buckets.empty:
                st.info("Još nema završenih modelskih poziva s token metapodacima.")
            else:
                st.line_chart(buckets.rename(columns={"input_tokens": "Ulazni", "output_tokens": "Izlazni"}))
    st.caption("Tokeni su iz metapodataka Ollame, zbrojeni za agent i security, uključujući ponovne pozive nakon alata. "
               "Pripisuju se trenutku završetka poziva, nisu prikaz stvarnog streaminga tokena. "
               "Sekundni graf prikazuje zadnjih 15 minuta s podacima iz odabranog razdoblja. Embedding tokeni nisu uključeni.")

    phases = events[events["event"].isin(["extraction_completed", "embedding_completed", "search_completed", "llm_completed", "upload_completed"])]
    if not phases.empty:
        st.subheader("Trajanja faza")
        table = phases.groupby("event")["duration_ms"].agg(["count", "median", lambda values: values.quantile(.95)])
        table.columns = ["Poziva", "Medijan (ms)", "p95 (ms)"]
        st.dataframe(table.round(2), width="stretch")
        st.caption("Faze se mogu preklapati: pretraga uključuje embedding upita. Ne zbrajaju se u ukupno trajanje.")
    calls = events[(events["event"] == "llm_completed") & (events["usage_available"] == True)]
    if not calls.empty:
        st.subheader("Potrošnja po modelu i ulozi")
        st.dataframe(calls.groupby(["model", "purpose"])[["input_tokens", "output_tokens"]].sum(), width="stretch")
    problems = events[events["level"].isin(["ERROR", "WARNING"])]
    if not problems.empty:
        st.subheader("Zadnje greške i odbijanja")
        columns = [name for name in ["timestamp", "event", "level", "path", "status_code", "error_type", "request_id"] if name in problems]
        st.dataframe(problems[columns].tail(50).iloc[::-1], hide_index=True, width="stretch")


render()
