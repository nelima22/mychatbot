import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


import streamlit as st
import os
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

def get_groq_api_key():
    # First, try to get from Streamlit secrets
    if 'GROQ_API_KEY' in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    
    # Then, try to get from environment variables
    if 'GROQ_API_KEY' in os.environ:
        return os.environ['GROQ_API_KEY']
    
    # If not found, return None
    return None


# Streamlit UI
st.set_page_config(page_title='Ask the Document App')
st.title('Ask the Document App')

st.header('About the App')
st.write("""
This is a question-answering platform that allows users to upload one or more PDF documents and receive answers to queries based on the content of the document(s). 

### How It Works
- Upload a Document: You can upload any PDF document(.pdf).
- Ask a Question: After uploading the document, type in your question related to the document's content.
- Get Answers: AI analyzes the document and provides answers based on the information contained in it.

## Get Started

""")
st.subheader("Get a Groq API key")
st.write("If you are pompted to enter a Groq API key that means there's an error on my end. Please input your Groq Api Key to proceed. If you don't have one you need to get your own API key by following the instructions:")
st.write("""
1. Go to the [Groq Console](https://console.groq.com/keys) and follow the prompts to get your API key.
2. Copy it and paste it in the field provided.        
3. Upload your document(s) and start asking questions.
""")


# Use the get_groq_api_key function to get the API key
groq_api_key = get_groq_api_key()

# If no API key is found, prompt the user
if not groq_api_key:
    st.warning("No Groq API key found in secrets or environment variables.")
    groq_api_key = st.text_input('Enter your Groq API Key:', type='password')
    if groq_api_key:
        # Store the entered key in session state
        st.session_state['groq_api_key'] = groq_api_key
    else:
        st.error("Please enter a Groq API key to proceed.")
        st.stop()


# Retrieve the key from session state if it was manually entered
if 'groq_api_key' in st.session_state:
    groq_api_key = st.session_state['groq_api_key']


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
