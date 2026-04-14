import urllib3
import requests
from googletrans import Translator, LANGUAGES
# Disable urllib3 warnings about SSL/TLS certificates (not recommended for production)
urllib3.disable_warnings()

from googletrans import Translator, LANGUAGES

text = "hi"
target_lang = "te"

def translate_text(text, target_lang):
    # Check if the target language code is valid
    if target_lang not in LANGUAGES:
        return "Invalid target language"
    # Translate the text to the target language
    try:
        translator = Translator(service_urls=["https://translate.google.com"])
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except Exception as e:
        return str(e)

k = translate_text(text, target_lang)
print(k)
