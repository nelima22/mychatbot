import streamlit as st
import os
import pypdf
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.llms import GroqModel

def generate_response(uploaded_file, query_text):
    if uploaded_file is not None and query_text:
        save_path = "./uploaded_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            data = source.fit(file_path, dtype="pdf", chunk_size=1024, chunk_overlap=0) 
            
            model_name='BAAI/bge-small-en-v1.5'
            embed_model = embeddings.HuggingFaceEmbeddings(model_name=model_name)

            llm = GroqModel(
                model='llama-3.1-70b-versatile', 
                groq_api_key=os.getenv('GROQ_API_KEY')
                )       
               
            retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)                   
            
            pipeline = generator.Generate(question=query_text, retriever=retriever, llm=llm)
            response = pipeline.call()        
            return response

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.error(f"Error type: {type(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return "Processing failed due to an error."
        
                    
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
        with st.spinner('Generating response...'):
            response = generate_response(uploaded_file, query_text)
            st.info(response)
else:
    st.warning("No file uploaded yet.")
