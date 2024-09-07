import streamlit as st
import tempfile
import os
from beyondllm import source, retrieve, llms, generator
from beyondllm.embeddings import FineTuneEmbeddings
from beyondllm.llms import GroqModel
from PyPDF2 import PdfReader
from io import BytesIO

# Function definitions
def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # Debug: check if file was written
            st.write("Temporary file created at:", temp_file_path)

            # Your existing code here
            data = source.fit(temp_file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
            llm = GroqModel(
                model='llama-3.1-70b-versatile',
                groq_api_key=os.getenv('GROQ_API_KEY') )
                 
            
            # Fine-tune embedding model
            fine_tuned_model = FineTuneEmbeddings()
            embed_model = fine_tuned_model.train(data, "BAAI/bge-small-en-v1.5", llm, "fintune")

            embed_model = fine_tuned_model.load_model("fintune")

            # Set up retriever
            retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)

            # Create QA pipeline
            pipeline = generator.Generate(question=query_text, retriever=retriever, llm=llm)        
            return pipeline.call()

        finally:
            # Clean up: remove the temporary file
            os.unlink(temp_file_path)

            
    return "No file uploaded or processing failed."


# Streamlit UI
st.set_page_config(page_title='Ask the Document App')
st.title('Ask the Document App')

st.header('About the App')
st.write("""
The App is an advanced question-answering platform that allows users to upload text documents and receive answers to their queries based on the content of these documents. Utilizing RAG approach powered by BeyondLLMs, the app provides insightful and contextually relevant answers.

### How It Works
- Upload a Document: You can upload any text document in .pdf format.
- Ask a Question: After uploading the document, type in your question related to the document's content.
- Get Answers: AI analyzes the document and provides answers based on the information contained in it.

# Get Started

""")
st.subheader("Get a Groq API key")
st.write("If you are pompted to enter a Groq API key and don't have one you can get your own API key by following the instructions:")
st.write("""
1. Go to the [Groq Console](https://console.groq.com/keys) and follow the prompts to get your API key.
2. Copy it and paste it in the field provided         
3. Upload your document and start asking questions
""")

# Try to get the API key from secrets or environment variables
groq_api_key = None

if 'GROQ_API_KEY' in os.environ:
    groq_api_key = os.environ['GROQ_API_KEY']
else:
    st.warning("No Groq API key found in secrets or environment variables. Please enter it below.")
    groq_api_key = st.text_input('Enter your Groq API Key:', type='password')
    if groq_api_key:
        groq_api_key = os.environ['GROQ_API_KEY']
    else:
        st.error("Please enter a Groq API key to proceed.")

# File upload
uploaded_file = st.file_uploader('Upload an article', type='pdf')

if uploaded_file:
    # Debug: confirm file upload
    st.write("File uploaded:", uploaded_file.name)

    # Query text
    query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')

    # Form input and query
    if st.button('Submit', disabled=not query_text):
        with st.spinner('Generating...'):
            response = generate_response(uploaded_file, query_text)
            st.info(response)
else:
    st.warning("No file uploaded yet.")
