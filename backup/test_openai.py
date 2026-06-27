from langchain_cohere import Cohere
llm = Cohere(model="command-r")

# Prueba simple
respuesta = llm.invoke("Hola, ¿puedes confirmar que la conexión funciona?")
print(respuesta)
