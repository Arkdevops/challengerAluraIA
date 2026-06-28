# Agente Cohere + LangChain

Este proyecto implementa un agente de consulta interactivo sobre documentos PDF utilizando LangChain, Cohere y HuggingFace.  
Permite realizar preguntas en lenguaje natural y obtener respuestas contextualizadas a partir del contenido del documento.

---

# Características
- Lectura automática de documentos PDF.
- División del texto en fragmentos para mejorar la búsqueda semántica.
- Generación de embeddings con HuggingFace.
- Almacenamiento en base vectorial con Chroma.
- Uso de modelos Cohere Chat (command-r-plus-08-2024, command-r-08-2024).
- Selección automática de perfil técnico o narrativo según la pregunta.
- Bucle interactivo: después de cada respuesta, vuelve a preguntar qué buscar.
- Configuración de claves mediante archivo .env.

---

# Estructura del proyecto

challengerAluraIA/
agentefinal.py        # Script principal del agente
guias/ERC_GES.pdf    # Carpeta con documentos PDF
db/                   # Base vectorial Chroma
.env                  # Variables de entorno (claves)

---

# Instalación
1. Clona el repositorio:
   ```bash
   git clone https://github.com/Arkdevops/challengerAluraIA/tree/main
   cd challengerAluraIA

2. Crea un archivo .env con tus claves:   
    COHERE_API_KEY=tu_clave_de_cohere
    HF_TOKEN=tu_token_de_huggingface

3. Ejecuta el agente:
    python agentefinal.py

    El programa te preguntará:

    ¿Qué quieres que busque en el documento? (o escribe 'salir' para terminar):

    Escribe tu consulta (ejemplo: tratamiento, diagnostico).

    El agente responderá y volverá a preguntarte.

    Escribe salir para finalizar la sesión.

    Ejemplo de interacción

    ¿Qué quieres que busque en el documento? (o escribe 'salir' para terminar): diagnostico.

    Respuesta:
    El diagnóstico o problema de salud mencionado en el texto es la Enfermedad Renal Crónica en su etapa 4 y 5.

    Requerimientos:

    langchain>=0.2.0
    langchain-core>=0.2.0
    langchain-huggingface>=0.0.1
    langchain-cohere>=0.6.0
    langchain-community>=0.2.0
    chromadb>=0.4.0
    sentence-transformers>=2.2.2
    pypdf>=4.0.0
    python-dotenv>=1.0.0
    cohere>=5.0.0

4. Notas
    langchain-huggingface: para usar HuggingFaceEmbeddings sin warnings.

    langchain-cohere: para el conector moderno ChatCohere.

    chromadb: base vectorial persistente.

    sentence-transformers: modelo all-MiniLM-L6-v2.

    pypdf: lectura de PDFs.

    python-dotenv: manejo de claves en .env.

    cohere: cliente oficial de Cohere.
