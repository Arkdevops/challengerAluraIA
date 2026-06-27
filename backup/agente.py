# --- Importaciones ---
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA
from pypdf import PdfReader

# --- Código principal ---
# 1. Leer el archivo PDF
pdf = PdfReader("guias/ER_Guia.pdf")
texto = ""
for pagina in pdf.pages:
    texto += pagina.extract_text()
print("Texto cargado")

# 2. Dividir en fragmentos
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = splitter.split_text(texto)
print(f"Se han generado {len(chunks)} fragmentos")

# 3. Crear embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Guardar vectores en Chroma
db = Chroma.from_texts(chunks, embeddings, persist_directory="db")
db.persist()
print("Base de datos guardada")

# 5. Definir perfiles Cohere
llm_tecnico = Cohere(model="command-r", temperature=0.2, max_tokens=400)
llm_narrativo = Cohere(model="command-r", temperature=0.8, max_tokens=700)

# 6. Función para elegir perfil según la pregunta
def elegir_llm(pregunta: str):
    tecnicas = ["procedimiento", "diagnóstico", "normativa", "protocolo", "urgencias", "tratamiento"]
    narrativas = ["historia", "relato", "tragedia", "emociones", "sentimientos", "narración"]

    if any(palabra in pregunta.lower() for palabra in tecnicas):
        return llm_tecnico
    elif any(palabra in pregunta.lower() for palabra in narrativas):
        return llm_narrativo
    else:
        return llm_tecnico  # por defecto

# 7. Hacer una consulta
pregunta = "¿Qué dice el documento sobre la atención en urgencias?"
llm_seleccionado = elegir_llm(pregunta)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm_seleccionado,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

respuesta = qa_chain.run(pregunta)

print("\nRespuesta completa:")
print(respuesta)
