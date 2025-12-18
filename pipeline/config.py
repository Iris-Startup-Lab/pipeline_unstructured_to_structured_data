# -*- coding: utf-8 -*-
## Iris Startup Lab 
## Fernando Dorantes Nieto
'''
<(*)
  ( >)"
  /|
'''
#### El config se usará para poder guardar las variables de ambiente
import os
from pathlib import Path 
from dotenv import load_dotenv

#env_path = Path(__file__).parent.parent / '.env'
env_path = Path(__file__).parent / '.env'

load_dotenv(dotenv_path=env_path, override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
SUPABASE_URL= os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_KEY_PSQL = os.getenv("SUPABASE_KEY_PSQL")
SUPABASE_URL_PSQL = os.getenv("SUPABASE_URL_PSQL")
SUPABASE_USER = os.getenv("SUPABASE_USER")
GITHUB_API_KEY = os.getenv("GITHUB_API_KEY")
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
PG_HOST = os.getenv('PG_HOST')
PG_PORT = os.getenv('PG_PORT')
PG_DBNAME = os.getenv('PG_DBNAME')
PG_USER = os.getenv('PG_USER')
PG_PASSWORD = os.getenv('PG_PASSWORD')