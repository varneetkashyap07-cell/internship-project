DocuMind AI PDF Chatbot
Executive Summary
DocuMind is an AI-powered PDF chatbot built in Python using Streamlit. It lets users upload or point to PDF documents and ask natural-language questions; the app extracts relevant content and returns answers via a conversational interface. Under the hood, DocuMind uses a large language model (e.g. Google’s Gemini or OpenAI’s API) to process document text. Its clean Streamlit UI provides an interactive chat window for the user. The executive summary should concisely describe this functionality and purpose. You can mention who created it (your name or team) and why (e.g. “to quickly query PDFs for answers”).
Features
PDF Chatbot: Users can query the content of PDF documents in plain English.
Conversational UI: Built with Streamlit’s st.chat_input and st.chat_message elements for a chat-like interface.
Multi-API Support: Integrates with LLM APIs (e.g. Google Gemini or OpenAI) via secure API keys.
Streamed Responses: The assistant’s answers appear in real-time as it generates them.
User-Friendly: Simple web app; no installation for end-users beyond prerequisites.
Extensible: Can be adapted to other document types or AI models.
Architecture & Tech Stack
DocuMind is written in Python (3.10–3.14) and leverages Streamlit for the front-end. It uses a virtual environment (venv) for dependency isolation. The AI core calls a language model API (either OpenAI or Google GenAI). We recommend using python-dotenv to manage API keys via a .env file. In Streamlit, we use st.chat_input for the user’s prompt box and st.chat_message to display chat bubbles. This creates a familiar chat UI where the user’s messages and the chatbot’s responses appear in a scrollable interface.
