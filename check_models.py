import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ No se encontró GEMINI_API_KEY en el archivo .env")
    exit(1)

genai.configure(api_key=api_key)

print("🔍 Consultando modelos disponibles para tu cuenta...\n")
try:
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            # Limpiamos el prefijo 'models/' para usarlo en LangChain
            nombre_limpio = model.name.replace('models/', '')
            print(f" - {nombre_limpio}")
except Exception as e:
    print(f"❌ Error de conexión: {e}")