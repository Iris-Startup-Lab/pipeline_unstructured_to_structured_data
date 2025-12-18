# -*- coding: utf-8 -*-
## Iris Startup Lab 
### Fernando Dorantes Nieto
### Version 1.0
'''
<(*)
  ( >)"
  /|
'''
import argparse
import ast
import sys
import os

# Asegurarse de que el directorio actual esté en el path para importar módulos locales
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline import main_pipeline

def parse_list(arg):
    """
    Parsea una cadena separada por comas o una lista literal de Python.
    Ejemplo entrada: "col1,col2,col3" -> ['col1', 'col2', 'col3']
    """
    if not arg:
        return []
    try:
        # Intenta evaluar como una lista de Python (ej: "['a', 'b']")
        return ast.literal_eval(arg)
    except (ValueError, SyntaxError):
        # Si falla, asume que es una cadena separada por comas
        return [x.strip() for x in arg.split(',')]

def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar el pipeline de procesamiento de datos no estructurados a estructurados."
    )

    # Argumentos obligatorios
    parser.add_argument("--folder_path", required=True, help="Ruta de la carpeta con archivos a procesar.")
    parser.add_argument("--examples_path", required=True, help="Ruta al archivo JSON de ejemplos.")
    parser.add_argument("--model_name", required=True, help="Nombre del modelo (ej. gemini-1.5-flash).")
    parser.add_argument("--prompt_for_extraction", required=True, help="Prompt base para la extracción.")
    parser.add_argument("--cols_order", required=False, type=parse_list, default=None, help="Lista de columnas ordenadas (separadas por coma).")
    parser.add_argument("--project_siglas", required=True, type=parse_list, help="Lista de siglas de proyectos (separadas por coma).")

    # Argumentos opcionales / Configuración de guardado
    parser.add_argument("--save_type", default="csv", choices=["csv", "txt"], help="Tipo de salida: 'csv' o 'txt'.")
    parser.add_argument("--base_folder_to_save_txts", help="Carpeta para guardar TXTs (requerido si save_type='txt').")
    parser.add_argument("--base_folder_to_save_csv", help="Carpeta para guardar CSV (requerido si save_type='csv').")
    parser.add_argument("--filename_to_save_csv", help="Nombre del archivo CSV de salida (requerido si save_type='csv').")
    
    # Otros
    parser.add_argument("--chunk_size_for_extraction", type=int, default=1000, help="Tamaño del chunk de texto.")

    args = parser.parse_args()

    # Validaciones de lógica de negocio
    if args.save_type == 'csv':
        missing = []
        if not args.base_folder_to_save_csv: missing.append("--base_folder_to_save_csv")
        if not args.filename_to_save_csv: missing.append("--filename_to_save_csv")
        
        if missing:
            parser.error(f"Para save_type='csv', faltan los siguientes argumentos: {', '.join(missing)}")

    if args.save_type == 'txt' and not args.base_folder_to_save_txts:
        parser.error("Para save_type='txt', se requiere --base_folder_to_save_txts")

    print("--- Iniciando Ejecución del Pipeline ---")
    print(f"Carpeta de entrada: {args.folder_path}")
    print(f"Modelo: {args.model_name}")
    
    # Pasar los argumentos como kwargs a la función principal
    main_pipeline(**vars(args))
    
    print("--- Ejecución finalizada ---")



if __name__ == "__main__":
    main()