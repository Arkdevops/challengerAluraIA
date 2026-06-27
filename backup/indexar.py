from pypdf import PdfReader

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("Leyendo PDF...")

pdf = PdfReader("guias/ER_Guia.pdf")

texto = ""

for pagina in pdf.pages:
    contenido = pagina.extract_text()

    if contenido:
        texto += contenido + "\n"

print("Dividiendo documento...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.create_documents([texto])

print(f"{len(docs)} fragmentos")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creando Chroma...")

db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="db"
)

print("Base vectorial creada correctamente.")