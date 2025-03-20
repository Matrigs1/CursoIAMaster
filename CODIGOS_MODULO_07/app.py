import os
import tempfile
import streamlit as st
from decouple import config
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

# Parte do RAG (LLM)
persist_directory = 'db'

# Essa função vai receber o arquivo que o usuário upou (vem em binário), cria um arquivo temporário no disco, escreve esse binário nele e, pega o path desse arquivo temporário.
# Após, iremos dar um load nesse arquivo como PDF e utilizar o text_splitter, para criar os chunks.
def process_pdf(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
        
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    os.remove(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400,
    )
    
    chunks = text_splitter.split_documents(documents=docs)
    return chunks

# Ao rodar a aplicação, se já existir arquivos persistidos (vector_store), vai dar load. Se não, retorna None.
def load_existing_vector_store():
    if os.path.exists(os.path.join(persist_directory)):
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(),
        )
        return vector_store
    return None

# Ao adicionar arquivos, se o vector_store já existir, dá o add_documents, senão, cria um vector_store e persite nele.
def add_to_vector_store(chunks, vector_store=None):
    if vector_store:
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=persist_directory,
        )
    return vector_store

def ask_question(model, query, vector_store):
    llm = ChatOpenAI(model=model)
    retriever = vector_store.as_retriever()
    
    system_prompt = '''
    Use o contexto para responder às pergunntas.
    Se não encontrar uma resposta no contexto, explique que não há informações disponíveis.
    Responda em formato markdown e com visualizações elaboaradas e interativas.
    Contexto: {context}
    '''
    
    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    
    response = chain.invoke({'input': query})
    return response.get('answer')

# Carregamos o vector_store para a aplicação.
vector_store = load_existing_vector_store()

st.set_page_config(
    page_title='Chat PyGPT',
    page_icon='📄',
)

st.header('🤖 Chat com seus documentos (RAG)')

with st.sidebar:
    st.header('Upload de arquivos 📄')
    uploaded_files = st.file_uploader(
        label='Faça upload de arquivos PDF',
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner('Processando documentos...'):
            all_chunks = []
            for uploaded_file in uploaded_files:
                chunks = process_pdf(file=uploaded_file)
                all_chunks.extend(chunks)
            
            vector_store = add_to_vector_store(
                chunks=all_chunks,
                vector_store=vector_store,
            )
            
    model_options = [
        'gpt-3.5-turbo',
        'gpt-4',
        'gpt-4-turbo',
        'gpt-4o-mini',
        'gpt-4o',
    ]
    
    selected_model = st.sidebar.selectbox(
        label='Selecione o modelo LLM',
        options=model_options,
    )
    
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    
question = st.chat_input('Como posso ajudar?')

if vector_store and question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))
    st.chat_message('user').write(question)
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.spinner('Buscando resposta...'):
        response = ask_question(
            model=selected_model,
            query=question,
            vector_store=vector_store,
        )
        st.chat_message('ai').write(response)
        st.session_state.messages.append({'role': 'ai', 'content': response})