from pypdf import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_chroma import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_cohere import ChatCohere

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


# ====================================================
# 1. Leer PDF
# ====================================================

pdf = PdfReader("guias/ER_Guia.pdf")

texto = ""

for pagina in pdf.pages:
    contenido = pagina.extract_text()
    if contenido:
        texto += contenido + "\n"

print("PDF cargado correctamente")


# ====================================================
# 2. Dividir documento
# ====================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.create_documents([texto])

print(f"Fragmentos creados: {len(chunks)}")


# ====================================================
# 3. Embeddings
# ====================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ====================================================
# 4. Crear base vectorial
# ====================================================

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db"
)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

print("Base vectorial creada")


# ====================================================
# 5. Modelos Cohere
# ====================================================

llm_tecnico = ChatCohere(
    model="command-r",
    temperature=0.2,
)

llm_narrativo = ChatCohere(
    model="command-r",
    temperature=0.8,
)


# ====================================================
# 6. Elegir modelo
# ====================================================

def elegir_llm(pregunta):

    tecnicas = [
        "procedimiento",
        "diagnóstico",
        "normativa",
        "protocolo",
        "urgencias",
        "tratamiento"
    ]

    narrativas = [
        "historia",
        "relato",
        "cuento",
        "emociones",
        "sentimientos"
    ]

    pregunta = pregunta.lower()

    if any(x in pregunta for x in tecnicas):
        return llm_tecnico

    if any(x in pregunta for x in narrativas):
        return llm_narrativo

    return llm_tecnico


# ====================================================
# 7. Prompt
# ====================================================

prompt = ChatPromptTemplate.from_template(
"""
Eres un experto analizando documentos.

Utiliza EXCLUSIVAMENTE el contexto entregado.

Si la respuesta no aparece en el documento responde:

"No encontré esa información en el documento."

Contexto:

{context}

Pregunta:

{input}
"""
)


# ====================================================
# 8. Consulta
# ====================================================

pregunta = "¿Qué dice el documento sobre la atención en urgencias?"

llm = elegir_llm(pregunta)

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

respuesta = rag_chain.invoke({
    "input": pregunta
})

print("\nRespuesta:\n")
print(respuesta["answer"])