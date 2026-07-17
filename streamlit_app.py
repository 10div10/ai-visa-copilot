"""
streamlit_app.py — Minimal chat UI for demoing the assistant locally / in interviews.

Run:
    streamlit run streamlit_app.py

(Requires the API to be running: uvicorn app.main:app --reload --port 8000)
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Visa Assistant", page_icon="🛂")
st.title("🛂 AI Visa Assistant")
st.caption("RAG over embassy rules + company policy · LangGraph · local embeddings + Groq")

with st.sidebar:
    st.header("Context")
    try:
        visa_types = requests.get(f"{API_URL}/visa-types", timeout=5).json()
    except Exception:
        visa_types = {}
        st.error("API not reachable. Start it with: uvicorn app.main:app --reload")

    visa_type = st.selectbox(
        "Visa type (for gap-analysis questions)",
        options=["(none)"] + list(visa_types.keys()),
        format_func=lambda x: visa_types.get(x, {}).get("label", x),
    )
    visa_type = None if visa_type == "(none)" else visa_type

    have_docs = []
    if visa_type:
        st.write("Documents you already have:")
        for doc_id in visa_types[visa_type]["documents"]:
            if st.checkbox(doc_id, key=doc_id):
                have_docs.append(doc_id)

    language = st.selectbox("Response language", ["English", "Hindi", "Spanish", "French", "German"])

    st.divider()
    st.header("Risk assessment (optional)")
    use_risk = st.checkbox("Include application details for risk scoring")
    application_details = None
    if use_risk:
        application_details = {
            "visa_type": visa_type or "us_h1b",
            "doc_completeness": st.slider("Document completeness", 0.0, 1.0, 0.8),
            "passport_buffer_days": st.number_input("Passport validity buffer (days)", value=180),
            "financial_ratio": st.slider("Financial proof (ratio of required min.)", 0.1, 3.0, 1.5),
            "ties_strength": st.slider("Ties to home country strength", 0.0, 1.0, 0.7),
            "days_to_travel": st.number_input("Days until travel/start date", value=45),
            "prior_rejection": 1 if st.checkbox("Prior visa rejection?") else 0,
        }
        st.caption("Try asking: \"How likely am I to get rejected and why?\"")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

query = st.chat_input("Ask about visa requirements, missing documents, timelines...")
if query:
    st.session_state.history.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "query": query,
                        "language": language,
                        "visa_type": visa_type,
                        "user_documents": have_docs,
                        "application_details": application_details,
                    },
                    timeout=60,
                )
                data = resp.json()
                answer = data["answer"]
                if data.get("risk_result"):
                    rr = data["risk_result"]
                    answer = (
                        f"**Risk: {rr['risk_level'].upper()} "
                        f"({rr['rejection_probability']:.0%} estimated rejection probability)**\n\n"
                        + answer
                    )
                if data.get("sources"):
                    answer += f"\n\n*Sources: {', '.join(data['sources'])}*"
            except Exception as e:
                answer = f"Error contacting API: {e}"
            st.markdown(answer)
    st.session_state.history.append(("assistant", answer))