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
import whisper
from pydub import AudioSegment
import speech_recognition as sr
import logging
import pandas as pd
import numpy as np

from video_to_audio import video_to_audio
# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


logging.info("Cargando modelo de Whisper 'medium'...")
whisperModel = whisper.load_model('medium')
logging.info("Modelo de Whisper cargado exitosamente.")

def transcribe_audio_array(audio_array: np.ndarray, language: str = "es") -> tuple[str, pd.DataFrame] | None:
    """
    Transcribe un array de audio NumPy utilizando el modelo Whisper.

    Args:
        audio_array (np.ndarray): El array de audio (float32) para transcribir.
                                  Debe estar a 16kHz y ser mono.
        language (str, optional): El idioma del audio. Por defecto es "es" (español).

    Returns:
        tuple[str, pd.DataFrame] | None: Una tupla con el texto completo y un DataFrame
                                        con los segmentos de la transcripción.
                                        Devuelve None si la operación falla.
    """
    if not isinstance(audio_array, np.ndarray):
        logging.error("La entrada no es un array de NumPy válido.")
        return None

    try:
        logging.info(f"Iniciando transcripción en idioma '{language}'...")
        # El array ya está en el formato que Whisper espera (float32)
        result = whisperModel.transcribe(audio_array, language=language)
        
        transcribed_text = result['text']
        #df_transcribed = pd.DataFrame(result['segments'])
        
        logging.info("Transcripción completada exitosamente.")
        return transcribed_text#, df_transcribed
    except Exception as e:
        logging.error(f"Error durante la transcripción con Whisper: {e}")
        return None


def transcribe_audio_file(audio_path: str, language: str = "es") -> str | None:
    """
    Transcribe un archivo de audio (ej. mp3, wav, m4a) a texto utilizando el modelo Whisper.

    Args:
        audio_path (str): La ruta al archivo de audio.
        language (str, optional): El idioma del audio. Por defecto es "es" (español).

    Returns:
        str | None: El texto transcrito si la operación es exitosa, en caso contrario, None.
    """
    if not os.path.exists(audio_path):
        logging.error(f"El archivo de audio no fue encontrado en la ruta: {audio_path}")
        return None

    try:
        logging.info(f"Iniciando transcripción del archivo '{os.path.basename(audio_path)}' en idioma '{language}'...")
        # Whisper puede procesar la ruta del archivo directamente
        result = whisperModel.transcribe(audio_path, language=language)
        transcribed_text = result['text']
        logging.info(f"Transcripción del archivo '{os.path.basename(audio_path)}' completada exitosamente.")
        return transcribed_text
    except Exception as e:
        logging.error(f"Error durante la transcripción del archivo '{audio_path}': {e}")
        return None



def transcribe_video_to_audio_to_text(video_path, language='es'):
    ### Crear archivo wav temporal 
    try:
        print('Creando archivo temporal')
        video_to_audio(video_path, 'temp.wav')
        audio_path = 'temp.wav'
        text_from_audio = transcribe_audio_file(audio_path=audio_path, language=language)
        print('Borrando archivo temporal')
        if os.path.isfile(audio_path):
            os.remove(audio_path)
        return text_from_audio
    except Exception as e: 
        logging.error(f"Error durante la transcripción del archivo '{video_path}': {e}")
        return None 
