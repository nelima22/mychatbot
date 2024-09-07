import streamlit as st
import tempfile
import os
from beyondllm import source, retrieve, llms, generator
from beyondllm.embeddings import FineTuneEmbeddings
from PyPDF2 import PdfReader
from io import BytesIO

# Google API Key input
google_api_key = st.text_input('Enter your Google API Key:', type='password')

# Setup environment
os.environ['GOOGLE_API_KEY'] = google_api_key

def generate_response(uploaded_file, google_api_key, query_text):
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Write the uploaded file content to the temporary file
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            # Use the temporary file path in source.fit()
            data = source.fit(temp_file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0)
               
            llm = llms.GeminiModel()

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


# Page title
st.set_page_config(page_title='Ask the Document App')
st.title('Ask the Document App')

# Explanation of the App
st.header('About the App')
st.write("""
The App is an advanced question-answering platform that allows users to upload text documents and receive answers to their queries based on the content of these documents. Utilizing RAG approach powered by BeyondLLMs, the app provides insightful and contextually relevant answers.

### How It Works
- Upload a Document: You can upload any text document in .pdf format.
- Ask a Question: After uploading the document, type in your question related to the document's content.
- Get Answers: AI analyzes the document and provides answers based on the information contained in it.


### Get Started
Simply upload your document and start asking questions!
""")

# Google API Key input
#google_api_key = st.text_input('Enter your Google API Key:', type='password')

# Setup environment
#os.environ['GOOGLE_API_KEY'] = google_api_key

if google_api_key:
    # File upload
    uploaded_file = st.file_uploader('Upload an article', type='pdf')
    
    if uploaded_file:
        # Query text
        query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.')
        
        # Form input and query
        if st.button('Submit', disabled=not query_text):
            with st.spinner('Calculating...'):
                response = generate_response(uploaded_file, google_api_key, query_text)
                st.info(response)
else:
    st.warning('Please enter your Google API Key to proceed.')

# Instructions for getting an API key
st.subheader("Get a Google API key")
st.write("You can get your own API key by following the instructions:")
st.write("""
1. Go to [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key).
2. Click on the Get an API key button then in the next page clivk on Create API Key buton and follow the prompts.
""")
