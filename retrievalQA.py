from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_cohere import Cohere

# 1. Cargar la base vectorial
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embeddings)

# 2. Crear el modelo Cohere
llm = Cohere(model="command-r")

# 3. Conectar el modelo con la base vectorial
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# 4. Hacer una consulta
pregunta = "¿Qué dice el documento sobre la atención en urgencias?"
respuesta = qa_chain.run(pregunta)

print("Respuesta completa:")
print(respuesta)
