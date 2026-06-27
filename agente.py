from Langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Crhoma
from pypdf import PdfReader

#1. leer el archivo PDF
pdf = PdfReader("guias/ERC_GES.pdf")
texto = ""
for pagina in pdf.pages:
    contenido = pagina.extract_text()
    if contenido:
        texto += contenido

print("Texto cargado")

#2. Dividir en fragmentos
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
chunks = splitter.split_text(texto)
print(f"Se han generado {len(chunks)} fragmentos")

#3. Crear embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#4. Guardar vectores en chroma
db = Chroma.from_texts(chunks, embeddings, persist_directory="db")
db.persist()
print("base de datos guardada")
