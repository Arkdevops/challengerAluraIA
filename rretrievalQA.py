from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.openai import OpenAI

# 1. Cargar la base vectorial
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embeddings)

# 2. Crear el modelo de lenguaje
llm = OpenAI(model_name="gpt-3.5-turbo")  # requiere tu API key

# 3. Conectar el modelo con la base vectorial
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# 4. Consulta en lenguaje natural
pregunta = "¿Qué dice el documento sobre la atención en urgencias?"
respuesta = qa_chain.run(pregunta)

print("Respuesta completa:")
print(respuesta)
