import os
from decouple import config
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

# Modelo da LLM.
model = ChatOpenAI(
    model='gpt-4o-mini',
)

# Pegando o arquivo do manual.
pdf_path = 'laptop_manual.pdf'
loader = PyPDFLoader(pdf_path)

# Dando load no arquivo com o objeto.
docs = loader.load()

# Criando um splitter com as configurações desejadas.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Fazendo de fato o split, passando o manual em pdf.
chunks = text_splitter.split_documents(
    documents=docs,
)

# Instanciando um objeto para embedding.
embedding = OpenAIEmbeddings()

# Criando um vector store com o Chroma e, passando um documento (com split) para ele.
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    collection_name='laptop_manual',
)

# Utilizando o vector store como retriever.
retriever = vector_store.as_retriever()

prompt = hub.pull('rlm/rag-prompt')

rag_chain = (
    {
        'context': retriever,
        'question': RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)

try:
    while True:
        question = input('Qual a sua dúvida?')
        response = rag_chain.invoke(question)
        print(response)
except KeyboardInterrupt:
    exit()
    