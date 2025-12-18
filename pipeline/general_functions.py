# -*- coding: utf-8 -*-
## Iris Startup Lab 
## Fernando Dorantes Nieto
'''
<(*)
  ( >)"
  /|
'''
##### Funciones generales 
import os 
import re 
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd 
import numpy as np 
import json 
from datetime import datetime, timezone


### Funciones de fecha 
#### Obtener la fecha actual
def get_current_date():
    current_iso_date = datetime.now().date().isoformat()
    return current_iso_date

#### Fecha actual pero en tiempo UTC
def get_current_timestamp_utc():
    utc_time = datetime.now(timezone.utc)
    return utc_time.isoformat()



### Funciones de archivos

def get_filelists_from_folder_root(folder_path, file_type):
    ### Función para detectar tipo de archivos
    #### Lo devuelve como lista 
    try:
        list_files_detected = [file for file in os.listdir(folder_path) if file.endswith(file_type)]
        return list_files_detected
    except Exception as e:
        print(f'Hay un error al obtener los archivos: {e}') 
        return []

def get_file_extension(file_path):
    file_path = Path(file_path)
    file_extension = file_path.suffix
    return file_extension



def get_filelists_from_folder(folder_path, file_type):
    ### Función para detectar tipo de archivos
    #### Lo devuelve como lista 
    list_files_detected = []
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(file_type):
                    list_files_detected.append(os.path.join(root, file))
        return list_files_detected
    except Exception as e:
        print(f'Hay un error al obtener los archivos: {e}') 
        return []

def get_unique_file_extensions(folder_path: str) -> List[str]:
    """
    Busca recursivamente y devuelve una lista de todas las extensiones de archivo
    únicas encontradas en una carpeta y sus subcarpetas.

    Args:
        folder_path (str): La ruta de la carpeta principal a explorar.

    Returns:
        List[str]: Una lista ordenada de extensiones de archivo únicas (ej. ['csv', 'pdf', 'txt']).
                   Devuelve una lista vacía si ocurre un error.
    """
    unique_extensions = set()
    try:
        for _, _, files in os.walk(folder_path):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext:  # Asegurarse de que el archivo tiene una extensión
                    unique_extensions.add(ext.lower().lstrip('.'))
        return sorted(list(unique_extensions))
    except Exception as e:
        print(f'Ocurrió un error al obtener las extensiones de archivo: {e}')
        return []




def analyze_string_project(filename_str, project_siglas): # No sé como se dice siglas en inglés XD
    if not filename_str.startswith(project_siglas):
        raise ValueError(f"Las siglas '{project_siglas}' no coinciden con el inicio de la cadena")
    
    split_str = filename_str.split('-')
    specific_name = split_str[-1]
    date_pattern = r'(\d{2})-(\d{2})'
    match_date = re.search(date_pattern, filename_str)
    
    if match_date:
        day = match_date.group(1)
        month = match_date.group(2)
        date_str = f"{day}/{month}"
    else:
        day = month = date_str = "Date not found"
    
    #pattern_file_version = r'(?<!-)(\d{4})(?!\d{2}[-])'
    pattern_file_version = r'[A-Za-z](\d+)(?=\d{2}-\d{2}-)'
    match_version = re.search(pattern_file_version, filename_str.replace(project_siglas, 
                                                                   '', 1))
    
    if match_version:
        version = match_version.group(1)
    else:
        version = "Version not found"
    
    string_extra_characters = filename_str[len(project_siglas):]
    pattern_job = r'^([A-Za-z]{2,4})'
    match_job = re.search(pattern_job, string_extra_characters)
    
    if match_job:
        kind_job = match_job.group(1)
    else:
        kind_job = "Job not found"
    
    
    return {
        'cadena_original': filename_str,
        'siglas_proyecto': project_siglas,
        'tipo_trabajo': kind_job,
        'dia': day,
        'mes': month,
        'fecha': day+'-' + month ,
        'numero_version': version,
        'nombre_especifico': specific_name
    }


def analyze_string_with_list(filename_str, list_siglas):
    siglas_found = []
    for siglas in list_siglas:
        if filename_str.startswith(siglas):
            siglas_found.append(siglas)
    
    if not siglas_found:
        raise ValueError(f'No se encontró ninguna sigla que coincida')

    project_siglas = max(siglas_found, key=len)
    return analyze_string_project(filename_str, project_siglas)



def get_file_category(filename):
    """
    Determina la categoría de un archivo (texto, audio, video, imagen)
    basándose en su extensión.

    Args:
        filename (str): El nombre del archivo (puede incluir la ruta completa).

    Returns:
        str: La categoría del archivo ("texto", "audio", "video", "imagen")
             o "desconocido" si la extensión no coincide con ninguna categoría.
    """
    _, ext = os.path.splitext(filename)
    ext = ext.lower().lstrip('.') 

    text_extensions = [
        'txt', 'csv', 'json', 'xml', 'md', 'log', 'py', 'html', 'css', 'js',
        'pdf', 'docx', 'xlsx', 'pptx', 'rtf', 'odt', 'ods', 'odp'
    ]
    audio_extensions = [
        'mp3', 'wav', 'aac', 'flac', 'ogg', 'm4a', 'wma', 'aiff', 'alac'
    ]
    video_extensions = [
        'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'mpeg', 'mpg', '3gp'
    ]
    image_extensions = [
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp', 'svg', 'ico', 'heif', 'heic'
    ]

    if ext in text_extensions:
        return "texto"
    elif ext in audio_extensions:
        return "audio"
    elif ext in video_extensions:
        return "video"
    elif ext in image_extensions:
        return "imagen"
    else:
        return "desconocido"



def get_filename(file_path):
    try:
        file_path = str(file_path)
        regex_to_get= r'(?!.*\\).+'
        x = re.search(regex_to_get, file_path)
        return x.group(0)
    except Exception as e: 
        print(f'Error detectado en la cadena {e}')
        return None 

def process_excel_file_filenames(filename_dir, sheet_names):
    try:
        list_dfs = []
        for sheet_name in sheet_names:
            df = pd.read_excel(filename_dir, sheet_name= sheet_name)
            columns_df = df.columns.tolist()
            columns_df = [re.sub(' ', '_', column.lower()) for column in columns_df]
            df.columns = columns_df
            list_dfs.append(df)
        return list_dfs[0], list_dfs[1], list_dfs[2]
    except Exception as e: 
        print(f'Error al procesar el archivo csv')
        return None, None, None 


def process_iris_excel_file(filename_dir, sheet_names):
    excel_file = filename_dir
    df_projects, df_work_type, df_clients = process_excel_file_filenames(excel_file, sheet_names = sheet_names)

    df_projects.columns = ['nombre_proyecto', 
                        'siglas_proyecto', 
                        'cliente', 
                        'fecha_inicio_proyecto',
                        'fecha_fin_proyecto']
    df_work_type.columns = ['siglas_tipo_trabajo', 
                            'descripcion_tipo_trabajo']
    df_clients.columns = ['cliente', 
                        'categoria_cliente']
    
    return df_projects, df_work_type, df_clients




def read_json_file(filename_str):

    with open(filename_str, 'r') as file:
        data = json.load(file)  

    return data 




def complete_dict(
    original_dict,
    list_keys,
    default_value='desconocido'):
    """
    Asegura que un diccionario contenga un conjunto de claves requeridas.

    Recorre una lista de claves. Si una clave existe en el diccionario original,
    mantiene su valor. Si no existe, la añade al nuevo diccionario con un
    valor por defecto.

    Args:
        diccionario_original (Dict[str, Any]): El diccionario de entrada que puede
                                               tener claves faltantes.
        claves_requeridas (List[str]): Una lista de strings con todas las claves
                                       que el diccionario final debe tener.
        valor_por_defecto (Any, optional): El valor que se asignará a las claves
                                           que no se encuentren. Por defecto es "desconocido".

    Returns:
        Dict[str, Any]: Un nuevo diccionario completo con todas las claves requeridas.
    """
    return {
        key: original_dict.get(key, default_value)
        for key in list_keys
    }

### load txt files
def load_text_files(directory):
    texts = {}
    for file in Path(directory).glob("*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:  
                texts[file.name] = content
    return texts

