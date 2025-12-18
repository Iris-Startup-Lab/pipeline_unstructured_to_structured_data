# -*- coding: utf-8 -*-
## Iris Startup Lab 
### Fernando Dorantes Nieto
### Version 1.0
'''
<(*)
  ( >)"
  /|
'''

import os
import json
import pandas as pd
import logging
import time 
import re 
from uuid import uuid4



# Para configuración y claves de API
from config import GEMINI_API_KEY

# --- Importar funciones de los módulos del proyecto ---

#### Comenzar con el pipeline general

from general_functions import (
    get_file_category, 
    get_filename, 
    analyze_string_project, 
    analyze_string_with_list, 
    load_text_files
)

# Para extracción de texto de diferentes formatos
from audio_to_text import transcribe_audio_file, transcribe_video_to_audio_to_text
from database_functions import read_tabular_data_to_string

from nlp_manipulation_and_functions import (
    convertDocument, 
    langextract_load_examples, 
    load_only_names_database, 
    langExtract_extract_values, 
    langExtract_value_to_df,
    chunk_text, 
    split_sentences
)

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_file_to_text(file_path: str) -> str | None:
    """
    Procesa un archivo para extraer su contenido de texto basado en su tipo.

    Args:
        file_path (str): La ruta al archivo a procesar.

    Returns:
        str | None: El texto extraído del archivo, o None si falla la extracción.
    """
    category = get_file_category(file_path)
    filename = os.path.basename(file_path)
    logging.info(f"Procesando archivo '{filename}' en la categoría: {category}")

    try:
        if category == "audio":
            return transcribe_audio_file(file_path)
        elif category == "video":
            return transcribe_video_to_audio_to_text(file_path)
        elif category == "texto":
            # convertDocument es para .docx, .pdf, etc.
            if file_path.endswith(('.csv', '.xlsx')):
                return read_tabular_data_to_string(file_path)
            else:
                return convertDocument(file_path)
        else:
            logging.warning(f"Categoría de archivo '{category}' no soportada para '{filename}'.")
            return None
    except Exception as e:
        logging.error(f"Error al procesar el archivo '{filename}': {e}")
        return None

def main_pipeline(folder_path: str, 
                  examples_path: str, 
                  model_name: str, 
                  #output_csv_path: str, 
                  prompt_for_extraction:str,
                  cols_order: list, 
                  project_siglas: list, 
                  base_folder_to_save_txts:str = None, 
                  base_folder_to_save_csv:str = None,
                  filename_to_save_csv:str = None,
                  save_type: str = 'csv',
                  chunk_size_for_extraction: int = 1000 
                  ):
    """
    Pipeline principal para procesar archivos, extraer datos y generar un CSV.

    Args:
        folder_path (str): Carpeta que contiene los archivos a procesar.
        examples_path (str): Ruta al archivo JSON con ejemplos para LangExtract.
        output_csv_path (str): Ruta donde se guardará el CSV final.
        project_siglas (str): Siglas del proyecto para analizar el nombre del archivo.
    """
    # --- Constantes y Configuración ---
    # 1. Cargar configuración para LangExtract
    logging.info("Cargando ejemplos y configuración para LangExtract...")
    with open(examples_path, 'r', encoding='utf-8') as f:
        json_examples = json.load(f)
    
    examples = langextract_load_examples(json_examples)
    columns_to_extract = load_only_names_database(json_examples)
    langextract_prompt = prompt_for_extraction
    model_id = model_name

    all_results_df = pd.DataFrame()
    concated_text = []
    # 2. Iterar sobre los archivos de la carpeta
    for root, _, files in os.walk(folder_path):
        conteo = 0
        for file in files:
            conteo += 1
            file_path = os.path.join(root, file)
            print(f'Procesando el siguiente archivo: \n {file_path}')
            # 3. Extraer texto del archivo
            text_content = process_file_to_text(file_path)

            if not text_content:
                logging.warning(f"No se pudo extraer texto de '{file}', saltando archivo.")
                continue

            if save_type=='csv':
                # 4. Dividir el texto en chunks y extraer valores de cada uno
                text_chunks = chunk_text(text_content, chunk_size=chunk_size_for_extraction)
                file_results_df = pd.DataFrame()

                logging.info(f"Extrayendo datos de {len(text_chunks)} chunks para el archivo '{file}'...")
                for i, chunk in enumerate(text_chunks):
                    logging.info(f"Procesando chunk {i+1}/{len(text_chunks)}...")
                    langextract_result = langExtract_extract_values(chunk, langextract_prompt, examples, GEMINI_API_KEY, model_id)

                    if not langextract_result or not langextract_result.extractions:
                        logging.warning(f"LangExtract no devolvió extracciones para el chunk {i+1} de '{file}'.")
                        continue
                    
                    df_chunk = langExtract_value_to_df(langextract_result, columns_to_extract)
                    file_results_df = pd.concat([file_results_df, df_chunk], ignore_index=True)

                # 5. Consolidar y añadir metadatos del archivo
                df_file = file_results_df # Ahora df_file contiene los resultados de todos los chunks
                
                try:

                    filename_info = analyze_string_with_list(get_filename(file_path), project_siglas)
                    for key, value in filename_info.items():
                        df_file[f"filename_{key}"] = value
                except ValueError as e:
                    logging.warning(f"No se pudo analizar el nombre del archivo '{file}': {e}")

                print('Imprimiendo los valores de dataframe')
                all_results_df = pd.concat([all_results_df, df_file], ignore_index=True)
                print(all_results_df.columns)
                time.sleep(60)
            elif save_type=='txt':
                filename_to_save_name = get_filename(file_path)
                filename_to_save_name = re.sub(r'^(.*)\.[^\.]+$', r'\1', filename_to_save_name)
                new_file_path_name = base_folder_to_save_txts + filename_to_save_name + '.txt'
                
                with open(new_file_path_name, "w") as file:
                    file.write(text_content)
                concated_text.append(text_content)
            
    if save_type=='csv':
    
        if cols_order:
            all_results_df = all_results_df[cols_order]
        all_results_df.to_csv(base_folder_to_save_csv + filename_to_save_csv, index=False)
        return all_results_df
    elif save_type=='txt': 
        return concated_text

#### ¿Quizás unir ambos pipeline?

def pipeline_texts_to_json(input_dir, 
                           output_dir_jsons,
                           output_dir_general_json, 
                           final_json_name, 
                           chunk_size, 
                           project_siglas):
   
    all_chunks = []
    file_texts = load_text_files(input_dir)
        
    for filename, text in file_texts.items():
        try:
            # 1. Analizar el nombre del archivo para obtener metadatos iniciales
            filename_info = analyze_string_with_list(filename, project_siglas)


        except ValueError as e:
            logging.warning(f"No se pudo analizar el nombre del archivo '{filename}': {e}. Saltando archivo.")
            continue

        chunks = chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "source_file": filename,
                "chunk_id": str(uuid4()),
                "chunk_index": i,
                "text": chunk, 
                **filename_info  # Desempaquetar el diccionario enriquecido
            }
            all_chunks.append(chunk_data)

            # Guardar cada chunk como JSON individual
            chunk_filename = f"{chunk_data['chunk_id']}.json"
            with open(os.path.join(output_dir_jsons, chunk_filename), "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    # Guardar archivo consolidado .jsonl
    filepath_name_general_json = output_dir_general_json + final_json_name + '.json'
    filepath_name_general_jsonl = output_dir_general_json + final_json_name + '.jsonl'
    
    # Guardar como un solo JSON array (no es lo mismo que JSONL)
    with open(filepath_name_general_json, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(filepath_name_general_jsonl, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
