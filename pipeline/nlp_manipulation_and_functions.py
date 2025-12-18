# -*- coding: utf-8 -*-
## Iris Startup Lab 
## Fernando Dorantes Nieto
'''
<(*)
  ( >)"
  /|
'''
import re 
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import numpy as np
from tabulate import tabulate
from docling.document_converter import DocumentConverter
from trafilatura import fetch_url, extract
from langextract import data
import langextract as lx
from transformers import AutoTokenizer
import json 
import dateparser
from spacy.matcher import Matcher
from spacytextblob.spacytextblob import SpacyTextBlob
from google import genai 
import tiktoken
from typing import Optional, Tuple
import pandas as pd 

from general_functions import complete_dict
from database_functions import generate_uuid


nltk.download('punkt_tab')
nlp = spacy.load("es_core_news_md")

nlp.add_pipe('spacytextblob')

#if 'spacytextblob' not in nlp.pipe_names:
#    nlp.add_pipe('spacytextblob')
#else:
#    nlp.add_pipe('spacytextblob')


def tokenize_text(text):
    return count_tokens(text, provider="huggingface", model_name="gpt2")



def get_gemini_available_models(client: genai.client.Client) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Obtiene los mejores modelos disponibles de Gemini por categoría ('pro', 'flash', 'gemma').

    Filtra los modelos para excluir versiones 'preview' y selecciona el que tiene el
    mayor límite de tokens de entrada para cada categoría.

    Args:
        client (genai.client.Client): El cliente de Google GenAI inicializado.

    Returns:
        Tuple[Optional[str], Optional[str], Optional[str]]: Una tupla con los nombres
        del mejor modelo 'pro', 'flash' y 'gemma', respectivamente.
        Si no se encuentra un modelo para una categoría, devuelve None para esa posición.
    """
    best_models = {
        'pro': {'name': None, 'limit': -1},
        'flash': {'name': None, 'limit': -1},
        'gemma': {'name': None, 'limit': -1}
    }

    try:
        all_models = client.models.list()
    except Exception as e:
        print(f"Error al listar los modelos de la API: {e}")
        return None, None, None

    for model in all_models:
        # Asegurarse de que el modelo tiene los atributos necesarios y no es una versión preview
        if not hasattr(model, 'input_token_limit') or 'preview' in model.name:
            continue

        model_name = model.name.replace('models/', '')
        token_limit = model.input_token_limit

        # Clasificar y actualizar el mejor modelo si el actual tiene más tokens
        if 'pro' in model_name and token_limit > best_models['pro']['limit']:
            best_models['pro'] = {'name': model_name, 'limit': token_limit}
        elif 'flash' in model_name and token_limit > best_models['flash']['limit']:
            best_models['flash'] = {'name': model_name, 'limit': token_limit}
        elif 'gemma' in model_name and token_limit > best_models['gemma']['limit']:
            best_models['gemma'] = {'name': model_name, 'limit': token_limit}

    return best_models['pro']['name'], best_models['flash']['name'], best_models['gemma']['name']


### DOCLING
### Función general con docling
def convertDocument(filename_string):
    converter = DocumentConverter()
    doc = converter.convert(filename_string).document
    md=  doc.export_to_markdown()
    return md 

def _normalize_text(text: str) -> str:
    """Función auxiliar para normalizar texto (minúsculas y sin acentos comunes)."""
    return text.lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')

### SPACY

def get_name_and_lastname(text: str) -> dict | None:
    """
    Extrae el primer nombre y apellido de una entidad de persona (PER) en el texto.

    Args:
        text (str): El texto a analizar.

    Returns:
        dict | None: Un diccionario con 'first_name' and 'last_name' si se encuentra,
                     o None si no se encuentra una persona.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PER':
            parts = ent.text.split()
            if len(parts) >= 2:
                return {"first_name": parts[0], "last_name": " ".join(parts[1:])}
            elif len(parts) == 1:
                return {"first_name": parts[0], "last_name": None}
    return None

def get_name(text: str) -> str | None:
    """
    Extrae el primer nombre de una entidad de persona (PER) en el texto.

    Args:
        text (str): El texto a analizar.

    Returns:
        str | None: El nombre si se encuentra, o None.
    """
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PER':
            return ent.text.split()[0]
    return None

def get_country(text: str) -> str | None:
    """
    Extrae una entidad geopolítica (GPE) que parece ser un país.

    Args:
        text (str): El texto a analizar.

    Returns:
        str | None: El nombre del país si se encuentra, o None.
    """
    doc = nlp(text)
    # spaCy a menudo etiqueta países como GPE (Geopolitical Entity)
    for ent in doc.ents:
        if ent.label_ == 'GPE':
            return ent.text
    return None

def get_mexico_state(text: str) -> str | None:
    """
    Busca y devuelve el nombre de uno de los 32 estados de México si se encuentra en el texto.

    Args:
        text (str): El texto a analizar.

    Returns:
        str | None: El nombre del estado mexicano encontrado, o None.
    """
    MEXICAN_STATES = [
    "aguascalientes", "baja california", "baja california sur", "campeche",
    "chiapas", "chihuahua", "ciudad de méxico", "coahuila",
    "colima", "durango", "estado de méxico", "guanajuato",
    "guerrero", "hidalgo", "jalisco", "méxico", "michoacán",
    "morelos", "nayarit", "nuevo león", "oaxaca",
    "puebla", "querétaro", "quintana roo", "san luis potosí",
    "sinaloa", "sonora", "tabasco", "tamaulipas",
    "tlaxcala", "veracruz", "yucatán", "zacatecas"
    ]
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'GPE' and _normalize_text(ent.text) in MEXICAN_STATES:
            return ent.text # Devuelve el texto original de la entidad
    return None

def standardize_date(
    text: str,
    gemini_client,
    api_key,
    gemini_model_name= "gemma-3-27b-it"
):
    """
    Detecta y estandariza una fecha a formato 'YYYY-MM-DD' usando varios métodos.

    Prueba en el siguiente orden:
    1.  dateparser: Intenta parsear el texto directamente (rápido y eficiente).
    2.  spaCy + dateparser: Usa el Matcher de spaCy para encontrar patrones de fecha
        comunes en español y luego los parsea.
    3.  Gemini: Como último recurso, usa un modelo de IA generativa para extraer
        y formatear la fecha.

    Args:
        text (str): El texto que contiene la fecha.
        gemini_client (genai.client.Client, optional): Cliente de Gemini inicializado.
                                                        Necesario para el método de IA.
        gemini_model_name (str, optional): Nombre del modelo de Gemini a usar.

    Returns:
        Optional[str]: La fecha en formato 'YYYY-MM-DD' o None si no se encuentra.
    """
    # --- Método 1: dateparser (El más directo y eficiente) ---
    try:
        print('Intentando con el método tradicional')
        parsed_date = dateparser.parse(text, languages=['es'])
        if parsed_date:
            print("Método 1 (dateparser) exitoso.")
            return parsed_date.strftime('%Y-%m-%d')
    except Exception:
        pass # Si falla, continuamos con el siguiente método

    # --- Método 2: spaCy Matcher + dateparser ---
    print('Intentando con el método Spacysoso')

    doc = nlp(text)
    matcher = Matcher(nlp.vocab)

    # Patrones comunes de fecha en español
    # 1. "15 de Enero de 2026"
    pattern1 = [{"IS_DIGIT": True}, {"LOWER": "de"}, {"POS": "PROPN"}, {"LOWER": "de"}, {"IS_DIGIT": True}]
    # 2. "15-01-2026" o "15/01/2026"
    pattern2 = [{"IS_DIGIT": True}, {"ORTH": {"IN": ["-", "/"]}}, {"IS_DIGIT": True}, {"ORTH": {"IN": ["-", "/"]}}, {"IS_DIGIT": True}]
    # 3. "Enero 15, 2026"
    pattern3 = [{"POS": "PROPN"}, {"IS_DIGIT": True}, {"ORTH": ","}, {"IS_DIGIT": True}]

    matcher.add("DatePatterns", [pattern1, pattern2, pattern3])
    matches = matcher(doc)

    if matches:
        # Usamos la coincidencia más larga encontrada
        first_match = sorted(matches, key=lambda m: m[2] - m[1], reverse=True)[0]
        _, start, end = first_match
        date_text = doc[start:end].text
        try:
            parsed_date = dateparser.parse(date_text, languages=['es'])
            if parsed_date:
                print("Método 2 (spaCy + dateparser) exitoso.")
                return parsed_date.strftime('%Y-%m-%d')
        except Exception:
            pass

    # --- Método 3: Gemini (IA Generativa) ---
    if gemini_client:
        print("Intentando con el Método 3 (Geminoso)...")
        client_gemini = gemini_client

        prompt = f"""
        Analiza el siguiente texto y extrae la fecha que encuentres.
        Devuelve la fecha únicamente en el formato YYYY-MM-DD.
        Si no encuentras ninguna fecha, responde solo con "None".

        Texto: "{text}"
        """
        try:
            response = client_gemini.models.generate_content(contents = prompt, model = gemini_model_name)
            result_text = response.text.strip()
            # Validar que la respuesta se parezca a una fecha y no sea "None"
            if result_text != "None" and re.match(r'\d{4}-\d{2}-\d{2}', result_text):
                print("Método 3 (Gemini) exitoso.")
                return result_text
        except Exception as e:
            print(f"Error con la API de Gemini: {e}")

    print("No se pudo estandarizar la fecha con ninguno de los métodos.")
    return None



def count_tokens(text: str, provider: str, model_name: str, client: Optional[genai.client.Client] = None) -> int:
    """
    Cuenta los tokens de un texto adaptándose a diferentes proveedores de modelos.

    Args:
        text (str): El texto a tokenizar.
        provider (str): El proveedor del modelo ('google', 'openai', 'huggingface').
        model_name (str): El nombre específico del modelo (ej. 'gemini-1.5-flash', 'gpt-4', 'bert-base-uncased').
        client (genai.client.Client, optional): Un cliente de Google GenAI ya inicializado.
                                                 Requerido si el proveedor es 'google'.

    Returns:
        int: El número de tokens.

    Raises:
        ValueError: Si el proveedor no es soportado o si falta el cliente para Google.
        Exception: Si ocurre un error durante la tokenización (ej. modelo no encontrado).
    """
    provider = provider.lower()

    try:
        if provider == 'google':
            if not client:
                raise ValueError("Se requiere un cliente de 'genai' para contar tokens con Google.")
            # La API de Google cuenta los tokens por nosotros
            result = client.models.count_tokens(model=f"models/{model_name}", contents=text)
            return result.total_tokens

        elif provider == 'openai':
            # Tiktoken es la librería oficial de OpenAI para tokenización
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))

        elif provider == 'huggingface':
            # Transformers usa AutoTokenizer para cargar el tokenizador correcto
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return len(tokenizer.encode(text))
        
        else:
            raise ValueError(f"Proveedor '{provider}' no soportado. Use 'google', 'openai' o 'huggingface'.")

    except Exception as e:
        print(f"Error al contar tokens para el modelo '{model_name}' del proveedor '{provider}': {e}")
        raise

### Funciones de split o chunk 
def chunk_text(text, chunk_size):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

#### Funciones de tokeninzación
##### Comenzando funciones
def split_sentences(df, text_column, columns_to_keep):
    """
    Divide el texto de una columna en frases y crea un nuevo DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        text_column (str): El nombre de la columna que contiene el texto a dividir.
        columns_to_keep (list): Una lista de nombres de columnas a mantener en el nuevo DataFrame.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con cada frase como una fila,
                      incluyendo las columnas especificadas y la columna de texto dividida.
    """
    new_rows = []
    for index, row in df.iterrows():
        sentences = sent_tokenize(row[text_column], language='spanish')
        for sent in sentences:
            new_row = {'text': sent.strip()}
            for col in columns_to_keep:
                new_row[col] = row[col]
            new_rows.append(new_row)
    return pd.DataFrame(new_rows)


#### AREA DE LANGEXTRACT 
def langextract_load_examples(json_items):
    list_out = []
    for item in json_items:
        ex_list = []
        for extract in item['extractions']:
            extracted_ = lx.data.Extraction(
                    extraction_class = extract['extraction_class'],
                    extraction_text=extract["extraction_text"],
                    attributes=extract.get("attributes", {}),
            )
            ex_list.append(extracted_)
        list_out.append(lx.data.ExampleData(text=item["text"], extractions=ex_list))
    return list_out


def load_only_names_database(json_items):
    list_out = []
    for item in json_items:
        for extract in item['extractions']:
            attributes=extract.get("attributes"),
            [list_out.append(x) for x in attributes[0].keys()]
    return list(set(list_out))


def langExtract_extract_values(text, prompt, examples, api_key, model_id):
    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            api_key=api_key,
            model_id=model_id
        )
        return result
    except Exception as e: 
        print(f"Error al obtener los valores {e}")
        return None

def langExtract_value_to_df(langextract_result, 
                            columns_selected, 
                            default_value='Desconocido'):
    list_extracted = []
    for extraction in langextract_result.extractions:
        extracted_data = extraction.attributes
        all_dict = complete_dict(original_dict=extracted_data, 
                                 list_keys=columns_selected, 
                                 default_value=default_value)
        all_dict['user_id'] = generate_uuid()

        # Contar cuántos valores son "Desconocido"
        unknown_count = sum(1 for value in all_dict.values() if value == default_value)
        total_columns = len(columns_selected)

        # Calcular el porcentaje de valores desconocidos y determinar la completitud
        if total_columns > 0:
            completeness_value = unknown_count / total_columns
        else:
            completeness_value = 1.0 # Si no hay columnas, está 0% completo

        all_dict['completeness_value'] = completeness_value
        # Se considera completo si el 50% o más de los campos están llenos (desconocidos <= 0.5)
        all_dict['completeness_bool'] = completeness_value <= 0.5

        # Añadir siempre el diccionario a la lista, sin importar la completitud.
        list_extracted.append(all_dict)

    return pd.DataFrame(list_extracted)
