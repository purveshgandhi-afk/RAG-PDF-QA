
import streamlit as st
import tempfile
from backend import setup_qa_system

st.set_page_config(page_title="RAG QA System", layout="centered")
CARD_STYLE = """
<style>
.rag-card {
  max-width: 900px;
  margin: 30px auto;
  border-radius: 16px;
  padding: 24px;
  background: linear-gradient(135deg, #e6f7ff 0%, #e8f9ee 100%);
  box-shadow: 0 6px 18px rgba(28, 57, 84, 0.08);
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
}
.header-title {
  text-align: center;
  margin-bottom: 6px;
  color: #0b6b3a;
}
.header-sub {
  text-align: center;
  margin-top: 0;
  margin-bottom: 18px;
  color: #0f5fa8;
}
.uploader {
  border-radius: 12px;
  padding: 10px;
  background: rgba(255,255,255,0.6);
}
.question-area {
  margin-top: 12px;
}
.answer-box {
  margin-top: 12px;
  background: #ffffff;
  padding: 12px;
  border-radius: 12px;
  border: 1px solid rgba(0,0,0,0.06);
}
.center-note {
  text-align:center;
  font-size: 13px;
  color: #235d3b;
  margin-top: 8px;
}
</style>
"""

st.markdown(CARD_STYLE, unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("<div class='rag-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='header-title'>RAG-based PDF QA System</h2>", unsafe_allow_html=True)
    st.markdown("<p class='header-sub'>Upload a PDF and ask questions about its contents</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="uploader")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
        st.session_state.pdf_path = None

    if uploaded_file is not None:
        # save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            temp_pdf_path = tmp.name

        if st.session_state.pdf_path != temp_pdf_path or st.session_state.qa_chain is None:
            with st.spinner("Processing PDF and building index (may take a moment)..."):
                try:
                    qa_chain = setup_qa_system(temp_pdf_path)
                    st.session_state.qa_chain = qa_chain
                    st.session_state.pdf_path = temp_pdf_path
                    st.success("PDF processed successfully. You can now ask questions.")
                except Exception as e:
                    st.session_state.qa_chain = None
                    st.error(f"Error processing PDF: {e}")

    else:
        st.info("Please upload a PDF to begin.")

    st.markdown("<div class='question-area'>", unsafe_allow_html=True)
    question = st.text_input("Enter your question here:", key="question_input")
    ask_btn = st.button("Ask", key="ask_btn")
    st.markdown("</div>", unsafe_allow_html=True)

    # Show answer if asked
    if ask_btn:
        if not st.session_state.get("qa_chain"):
            st.warning("No PDF processed yet. Upload a PDF first.")
        elif not question or question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Fetching answer..."):
                try:
                    answer = st.session_state.qa_chain.run(question)
                except Exception as e:
                    answer = f"Error obtaining answer: {e}"
            st.markdown(f"<div class='answer-box'><strong>Answer:</strong><div style='margin-top:6px'>{answer}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='center-note'>Minimal UI — upload PDF on the left and ask questions here.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
