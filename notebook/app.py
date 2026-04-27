import streamlit as st

st.set_page_config(
    page_title="Gmail Retrieval Assistant",
    layout="centered"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}

.header-card {
    padding: 14px 18px;
    border-radius: 10px;
    background-color: #f8f9fb;
    border: 1px solid #e6e8eb;
    margin-bottom: 10px;
}

.title {
    font-size: 24px;
    font-weight: 600;
}

.subtitle {
    color: #6b7280;
    font-size: 14px;
}

.divider {
    height: 1px;
    background-color: #eceff1;
    margin: 10px 0 20px 0;
}

.footer {
    text-align: center;
    color: #9aa0a6;
    font-size: 12px;
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
<div class="title">📬 Gmail Retrieval Assistant</div>
<div class="subtitle">Search and explore your emails using semantic retrieval</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask something about your emails...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving information..."):
            response = "This is a UI preview response. Backend will be connected later."
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown(
    '<div class="footer">Local Gmail RAG • Streamlit UI Preview</div>',
    unsafe_allow_html=True
)