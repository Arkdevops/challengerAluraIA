from pypdf import PdfReader

archivo = "guias/ERC_GES.pdf"

pdf = PdfReader(archivo)

texto = ""

for pagina in pdf.pages:
    contenido = pagina.extract_text()
    
    if contenido:
        texto += contenido
        
print("Documento cargado")
print()

print("Cantidad de páginas: ", len(pdf.pages))
print("Cantidad de caracteres: ", len(texto))   

     