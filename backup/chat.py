import os

from dotenv import load_dotenv

from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_cohere import ChatCohere

from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

llm = ChatCohere(
    model="command-r",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_template("""
Eres un experto analizando documentos.

Contesta únicamente usando el contexto.

Si no sabes la respuesta responde:

"No encontré esa información."

Contexto:

{context}

Pregunta:

{question}
""")

print("Chat iniciado.\n")

while True:

    pregunta = input("\nPregunta: ")

    if pregunta.lower() == "salir":
        break

    docs = retriever.invoke(pregunta)

    contexto = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    mensajes = prompt.invoke({
        "context": contexto,
        "question": pregunta
    })

    respuesta = llm.invoke(mensajes)

    print("\nRespuesta:\n")

    print(respuesta.content)