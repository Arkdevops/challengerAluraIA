# --- Importaciones ---
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

# --- Verificación de claves ---
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not cohere_api_key:
    raise ValueError("⚠️ No se encontró la clave de Cohere en .env")
else:
    print("Clave de Cohere detectada correctamente")

# --- Leer PDF ---
pdf = PdfReader("guias/ERC_GES.pdf")
texto = "".join([p.extract_text() for p in pdf.pages])
print("Texto cargado")

# --- Dividir en fragmentos ---
splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(texto)
print(f"Se han generado {len(chunks)} fragmentos")

# --- Embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Vector ---
db = Chroma.from_texts(chunks, embeddings, persist_directory="db")
print("Base de datos guardada")

# --- Modelos Cohere ---
llm_tecnico = ChatCohere(model="command-r-plus-08-2024", temperature=0.2, max_tokens=400)
llm_narrativo = ChatCohere(model="command-r-08-2024", temperature=0.8, max_tokens=700)

def elegir_llm(pregunta: str):
    tecnicas = ["procedimiento", "diagnostico", "normativa", "protocolo", "urgencias", "tratamiento"]
    narrativas = ["historia", "relato", "tragedia", "emociones", "sentimientos", "narración"]
    if any(p in pregunta.lower() for p in tecnicas):
        return llm_tecnico
    elif any(p in pregunta.lower() for p in narrativas):
        return llm_narrativo
    return llm_tecnico

# --- Bucle interactivo de consultas ---
while True:
    pregunta = input("\n¿Qué quieres que busque en el documento? (o escribe 'salir' para terminar): ")

    if pregunta.lower() in ["salir", "exit", "quit"]:
        print("👋 Finalizando agente. ¡Hasta pronto!")
        break

    llm_seleccionado = elegir_llm(pregunta)
    docs = db.similarity_search(pregunta, k=3)

    prompt = ChatPromptTemplate.from_template(
        "Usa la siguiente información del documento para responder la pregunta:\n\n{context}\n\nPregunta: {input}"
    )

    chain = prompt | llm_seleccionado | StrOutputParser()
    contexto = "\n\n".join([doc.page_content for doc in docs])
    respuesta = chain.invoke({"context": contexto, "input": pregunta})

    print("\nRespuesta:")
    print(respuesta)