from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
# Cargar la base vectorial
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embeddings)

# Hacer una consulta
pregunta = "¿Qué dice el documento sobre la atención en urgencias?"
resultados = db.similarity_search(pregunta, k=3)

print("Respuesta aproximada:")
for r in resultados:
    print(r.page_content[:300])  # muestra un fragmento
