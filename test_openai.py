from langchain_openai import OpenAI

# Inicializar el modelo
llm = OpenAI(model="gpt-3.5-turbo")

# Prueba simple
respuesta = llm.invoke("Hola, ¿puedes confirmar que la conexión funciona?")
print(respuesta)
