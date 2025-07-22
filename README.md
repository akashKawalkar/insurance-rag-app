# 🛡️ Insurance RAG Assistant

An AI-powered insurance policy assistant that provides accurate answers by retrieving information directly from uploaded insurance documents. Built using **FastAPI**, **Streamlit**, and state-of-the-art retrieval and large language models.

---

## 🚀 Features

- 💬 Ask questions about your insurance policies with contextual understanding
- 🔎 Retrieves answers from your uploaded insurance policy documents (avoids hallucinations)
- 🧠 Uses custom retrieval and reranking pipelines with large language models for precise answers
- 🖥️ Interactive frontend powered by Streamlit
- ⚙️ Robust backend API built with FastAPI

---

## 🛠️ Installation & Usage (Local)

✅ **Recommended for local testing and development**

### 1. Clone the Repository
git clone https://github.com/akashKawalkar/insurance-rag-app.git
cd insurance-rag-app

### 2. Set Your Environment Variables
Create a `.env` file in the project root directory with necessary secrets/keys (if any).

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Backend API
uvicorn app.main:app --reload --port 8000

### 5. Run the Frontend App
In a new terminal:
streamlit run app/frontend_app.py


### 6. Access the Application

Open your browser at: [http://localhost:8501](http://localhost:8501/)

---

## 📚 Model & Stack

| Component      | Tool / Framework / Model            |
|----------------|------------------------------------|
| Backend API    | FastAPI                            |
| Frontend       | Streamlit                          |
| Retrieval      | Custom hybrid search + reranking   |
| Language Model | HuggingFace Transformers (custom)  |
| Deployment     | Local development (Docker optional)|

---

## 🛡️ Disclaimer

This app is meant for informational purposes and does not replace professional insurance advice. Always review your official policy documents or consult with a licensed agent.

---

## 📜 License

This project is licensed under the Apache License.

---

## 📌 Created by Akash Kawalkar






