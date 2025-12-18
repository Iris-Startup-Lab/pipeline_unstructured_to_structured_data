# Pipeline: Datos no estructurados a estructurados
![icono](./icons/icon_pipeline.png)



El objetivo de este repositorio es de compartir el pipeline principal
del proyecto de usuarios sintéticos, el cuál es un proyecto interno  de Iris Startup Lab.

No obstante, este pipeline está diseñado para cualquier caso de uso usando este código.

Este pipeline es un desarrollo simplificado para su fácil ejecución y lectura


## Datos de ejemplo
Se han generado algunos archivos de ejemplo sobre entrevistas a tianguistas
Como es información pública, se han guardado en una carpeta de [Google Drive](https://drive.google.com/)

Donde pueden descargar los archivos de los cuáles se les va a extraer la información

## Modelos de lenguaje usados
En este caso se requiere una api de Gemini u OpenAI o modelos locales instalados para su uso
### ¿Cómo obtener una clave de API o usar un modelo local?
- Gemini: [AIStudio](https://aistudio.google.com/)
- OpenAi: [OpenAI Api Key](https://platform.openai.com/account/api-keys)
- Modelos locales: 
    - [HuggingFace](https://huggingface.co/models)
    - [Ollama](https://ollama.com/)  
    - [LM Studio](https://lmstudio.ai/)

### Modelos recomendados
- Gemini: 'gemini-2.5-flash-lite' Eficiente y rápido, además de que permite muchas llamadas
- OpenAi: 'chatgpt-4.2-32k'
- Modelos locales:
    - HuggingFace: 'DeepSeek r1:8B'
    - Ollama: 'Llama 3.2'  Algo pesado, necesitarás una GPU para su ejecución
    - LM Studio: 'DeepSeek r1:8B' 


## Modelos pequeños usados
- Este código usa [Whisper](https://github.com/openai/whisper) para poder transcribir audio a texto
- Se usan técnicas de procesamiento de lenguaje natural usando [Spacy](https://spacy.io/models/es)  y sus modelos en español

## Librerías clave
[Docling](https://docling-project.github.io/docling/) fue la librería clave que ayuda a convertir cualquier documento de texto a Markdown.

## Requerimientos técnicos mínimos
    - Python >= 3.10
    - RAM >= 16 GB
    - GPU (deseable), se pueden usar las de Colaboratory gratuitamente

## Cómo ejecutar el pipeline
**Pasos para ejecutar la DEMO:**

1. Instalar Python >= 3.10
    Puedes utilizar las siguientes opciones:
    [Anaconda](https://anaconda.org/)
    [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    O puedes utilizar versiones de Python online:
    [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
Nota: De preferencia usar Linux o [WSL](https://learn.microsoft.com/es-es/windows/wsl/install)
2. Instalar los paquetes dentro del archivo requirements.txt y guardar los secretos en un archivo nuevo llamado .env
3. Crear una carpeta llamada "data" y guardar todos los documentos que están alojados en esta carpeta en [Google Drive](https://drive.google.com/drive/folders/1b86llzTkWC6aFGwkbr_DA9iuZXdl-qXw?usp=sharing).
4. Cambiar la estructura del archivo de ejemplos en el archivo llamado `examples_for_langextract.json`, esto debe ir con los datos que se deben de extraer.
5. Ejecutar en PowerShell o en consola en base al archivo `python main_execute_pipeline.py`



**Pasos para cualquier propósito:** 

1. Instalar Python >= 3.10
    Puedes utilizar las siguientes opciones:
    [Anaconda](https://anaconda.org/)
    [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
    O puedes utilizar versiones de Python online:
    [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)
Nota: De preferencia usar Linux o [WSL](https://learn.microsoft.com/es-es/windows/wsl/install)
2. Instalar los paquetes dentro del archivo requirements.txt y guardar los secretos en un archivo nuevo llamado .env
3. Crear una carpeta llamada "data" y guardar todos los documentos de los cuáles extraerás información
4. Cambiar la estructura del archivo de ejemplos en el archivo llamado `examples_for_langextract.json`, esto debe ir con los datos que se deben de extraer.
5. Ejecutar en PowerShell o en consola: `python main_execute_pipeline.py`
 
## Ejemplos de ejecución

### Ejemplo 1: Generar salida en CSV
Este comando (bash) procesa los archivos en `data/`, usa el modelo `gemini-1.5-flash` y guarda los resultados en un CSV. Utiliza el nombre del archivo fuente como identificador.

```bash
python main_execute_pipeline.py \
  --folder_path "./data" \
  --examples_path "./examples.json" \
  --model_name "gemini-1.5-flash" \
  --prompt_for_extraction "Extrae los datos principales del contrato" \
  --cols_order "Fecha,Proveedor,Monto,Concepto" \ ## No es necesario
  --project_siglas "PRJ-A, PRJ-B" \
  --save_type "csv" \
  --base_folder_to_save_csv "./output" \
  --filename_to_save_csv "extraccion.csv"
```

### Ejemplo 2: Generar salida en TXT
Este comando (bash) extrae información y guarda archivos de texto individuales en la carpeta especificada.

```bash
python main_execute_pipeline.py \
  --folder_path "./data" \
  --examples_path "./examples.json" \
  --model_name "gemini-1.5-flash" \
  --prompt_for_extraction "Resume el contenido" \
  --cols_order "Resumen" \ ## No es necesario
  --project_siglas "GENERAL" \
  --save_type "txt" \
  --base_folder_to_save_txts "./output_txt"
```

## Autor:
Fernando Dorantes Nieto (AI Engineer/Data Scientist)

Iris Startup Lab

Cualquier duda o comentario favor de crear un "issue" o un "pull request" dentro
del repositorio.



## Ejemplo de vista:

En esta carpeta de [Google Drive](https://drive.google.com/drive/folders/1b86llzTkWC6aFGwkbr_DA9iuZXdl-qXw?usp=sharing) pueden ver los archivos que serán convertidos

En esta hoja de cálculo de Google Sheets pueden ver un ejemplo de como se ven los datos ya extraídos


