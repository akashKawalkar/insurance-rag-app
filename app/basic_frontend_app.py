import streamlit as st
import requests

st.set_page_config(page_title="Insurance Q&A App", page_icon=":shield:", layout="centered")

st.title("🛡️ Insurance Policy Assistant")
st.write("Ask any question about your insurance policy.")

# User input
query = st.text_area("Enter your question:", height=100)
submit = st.button("Get Answer")

if submit and query.strip():
    with st.spinner("Retrieving answer..."):
        try:
            response = requests.post(
                "http://localhost:8000/api/answer",
                json={"query": query},
                timeout=120
            )
            if response.status_code == 200:
                resp_json = response.json()
                st.markdown("### **Answer:**")
                st.success(resp_json.get("answer", "No answer returned."))

                # Show more details if desired
                if resp_json.get("top_chunks"):
                    with st.expander("Show supporting document excerpts"):
                        for i, chunk in enumerate(resp_json["top_chunks"], 1):
                            st.markdown(f"**Chunk #{i} (Score: {chunk.get('score'):.3f}):**\n\n{chunk['chunk']}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Could not get response from backend: {e}")

elif submit:
    st.warning("Please enter a question.")

st.markdown("---")
st.info("Make sure your FastAPI backend is running at [http://localhost:8000](http://localhost:8000).")

