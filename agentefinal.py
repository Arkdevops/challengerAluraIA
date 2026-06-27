# --- Importaciones ---
import os
from dotenv import load_dotenv
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pypdf import PdfReader

# --- Verificación de clave Cohere ---
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

if not cohere_api_key:
    raise ValueError(
        "⚠️ No se encontró la clave de Cohere. "
        "Por favor, crea un archivo .env en la carpeta del proyecto con la línea:\n"
        "COHERE_API_KEY=tu_clave_aqui"
    )
else:
    print("✅ Clave de Cohere detectada correctamente")

# --- Código principal ---
# 1. Leer el archivo PDF
ruta_pdf = "guias/ERC_GES.pdf"
pdf = PdfReader(ruta_pdf)
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

# 5. Definir perfiles Cohere con Chat API
llm_tecnico = ChatCohere(model="command-r-plus-08-2024", temperature=0.2, max_tokens=400)
llm_narrativo = ChatCohere(model="command-r-08-2024", temperature=0.8, max_tokens=700)

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

# 8. Recuperar documentos relevantes
docs = db.similarity_search(pregunta, k=3)

# 9. Crear prompt y cadena manualmente (nuevo enfoque)
prompt = ChatPromptTemplate.from_template(
    "Usa la siguiente información del documento para responder la pregunta:\n\n{context}\n\nPregunta: {input}"
)

chain = (
    prompt
    | llm_seleccionado
    | StrOutputParser()
)

contexto = "\n\n".join([doc.page_content for doc in docs])
respuesta = chain.invoke({"context": contexto, "input": pregunta})

print("\nRespuesta completa:")
print(respuesta)
