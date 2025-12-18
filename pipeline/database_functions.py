# -*- coding: utf-8 -*-
## Iris Startup Lab 
## Fernando Dorantes Nieto
'''
<(*)
  ( >)"
  /|
'''
import os 
import re
import uuid
import numpy as np
import pandas as pd 
from datetime import datetime, timezone 
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timezone
import logging
from config import PG_HOST, PG_PORT, PG_DBNAME, PG_USER, PG_PASSWORD


### Función para UUID
def generate_uuid():
    uuid4 = uuid.uuid4()
    return uuid4


### Funciones para las bases de datos de nombres clave


#### Funciones para datos tabulares
def clean_tabular_data(df):
    """
    Limpia un DataFrame convirtiendo inteligentemente las columnas a los tipos de datos más apropiados.

    - Detecta y convierte columnas de tipo objeto (texto) que parecen fechas al formato 'YYYY-MM-DD'.
    - Convierte columnas de tipo objeto que parecen numéricas a enteros o flotantes.

    Args:
        df (pd.DataFrame): El DataFrame de entrada para limpiar.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con las columnas limpiadas y convertidas.
    """
    df_cleaned = df.copy()

    for col in df_cleaned.columns:
        # Solo procesar columnas de tipo 'object', que es donde suelen estar los datos "sucios".
        if df_cleaned[col].dtype == 'object':
            # 1. Intentar convertir a datetime
            try:
                # `errors='coerce'` convierte los no-fechas en NaT (Not a Time)
                temp_col = pd.to_datetime(df_cleaned[col], errors='coerce')
                
                # Si un porcentaje significativo (ej. >75%) se convirtió a fecha, asumimos que es una columna de fecha
                if temp_col.notna().sum() / len(df_cleaned[col].dropna()) > 0.75:
                    # Formatear a YYYY-MM-DD. Esto la convierte de nuevo a objeto (string), pero con el formato deseado.
                    df_cleaned[col] = temp_col.dt.strftime('%Y-%m-%d')
                    continue # Pasar a la siguiente columna
            except (ValueError, TypeError):
                # No se pudo convertir a fecha, continuar al siguiente intento
                pass
            numeric_col = pd.to_numeric(df_cleaned[col], errors='coerce')
            
            if numeric_col.notna().any(): # Si al menos un valor se pudo convertir
                if (numeric_col.dropna() % 1 == 0).all():
                    df_cleaned[col] = numeric_col.astype('Int64')
                else:
                    df_cleaned[col] = numeric_col 
    return df_cleaned

def read_tabular_data_to_string(file_path, sheet_name=None, 
                                sep = ','):
    if file_path.endswith('.csv') or file_path.endswith('.tsv') :
        df = pd.read_csv(file_path, sep=sep)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        SystemError('Not supported file type')
    df = clean_tabular_data(df).dropna()
    return df.to_string()


def read_tabular_data_to_df(file_path, sheet_name=None, 
                                sep = ','):
    if file_path.endswith('.csv') or file_path.endswith('.tsv') :
        df = pd.read_csv(file_path, sep=sep)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        SystemError('Not supported file type')
    df = clean_tabular_data(df)
    df = df.dropna()
    return df


## Funciones de supabase
def pandas_dtype_to_pg(dtype):
    """Convierte un tipo de dato de Pandas a un tipo de dato PostgreSQL."""
    if pd.api.types.is_integer_dtype(dtype): return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype): return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype): return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype): return "TIMESTAMP WITHOUT TIME ZONE"
    else: return "TEXT"

def get_create_table_sql(df_for_schema: pd.DataFrame, qualified_table_name: str, has_input_reference_col: bool, has_session_id_col: bool) -> str:
    """Genera una sentencia SQL CREATE TABLE IF NOT EXISTS a partir de un DataFrame."""
    columns_defs = ["id UUID PRIMARY KEY DEFAULT gen_random_uuid()"]
    if has_input_reference_col: columns_defs.append("input_reference TEXT")
    if has_session_id_col: columns_defs.append("session_id TEXT")
    for col_name, dtype in df_for_schema.dtypes.items():
        pg_type = pandas_dtype_to_pg(dtype)
        columns_defs.append(f'"{col_name}" {pg_type}')
    return f"""CREATE TABLE IF NOT EXISTS {qualified_table_name} (\n    {',\n    '.join(columns_defs)}\n);"""


logger = logging.getLogger(__name__)

def _get_db_connection():
    """Establece y devuelve una conexión a la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, dbname=PG_DBNAME,
            user=PG_USER, password=PG_PASSWORD
        )
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"No se pudo conectar a la base de datos PostgreSQL: {e}")
        return None


def save_df_to_db(df: pd.DataFrame, table_name: str, schema: str = 'public', conflict_column: str = 'user_id'):
    """
    Guarda un DataFrame en una tabla de PostgreSQL, realizando una operación de "upsert".
    Inserta nuevas filas o actualiza las existentes en caso de conflicto en la columna especificada.

    Args:
        df (pd.DataFrame): El DataFrame a guardar.
        table_name (str): El nombre de la tabla de destino.
        schema (str, optional): El esquema de la base de datos. Por defecto es 'public'.
        conflict_column (str, optional): La columna con una restricción única (como una clave primaria)
                                         para determinar conflictos. Por defecto es 'user_id'.
    """
    if df.empty:
        logger.info("DataFrame está vacío, no se realizará ninguna operación en la base de datos.")
        return

    if conflict_column not in df.columns:
        logger.error(f"La columna de conflicto '{conflict_column}' no se encuentra en el DataFrame. Abortando.")
        return

    conn = _get_db_connection()
    if not conn:
        logger.error("No se pudo establecer conexión con la base de datos.")
        return

    try:
        with conn.cursor() as cursor:
            df_insert = df.copy()
            # Limpiar valores NaN/NaT para compatibilidad con la base de datos
            df_insert.replace({pd.NaT: None, np.nan: None}, inplace=True)

            # Crear la tabla si no existe
            qualified_table_name = f'"{schema}"."{table_name}"'
            
            # Generar la sentencia CREATE TABLE dinámicamente
            column_defs = [f'"{col}" {pandas_dtype_to_pg(dtype)}' for col, dtype in df_insert.dtypes.items()]
            column_defs.append(f'CONSTRAINT {table_name}_pkey PRIMARY KEY ("{conflict_column}")')
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {qualified_table_name} (\n    " + ",\n    ".join(column_defs) + "\n);"
            cursor.execute(create_table_sql)

            # Preparar la sentencia de UPSERT
            cols = df_insert.columns
            cols_sql = ", ".join([f'"{col}"' for col in cols])
            
            # Columnas a actualizar en caso de conflicto (todas excepto la clave de conflicto)
            update_cols = [col for col in cols if col != conflict_column]
            update_sql = ", ".join([f'"{col}" = EXCLUDED."{col}"' for col in update_cols])

            insert_stmt = (
                f'INSERT INTO {qualified_table_name} ({cols_sql}) VALUES %s '
                f'ON CONFLICT ("{conflict_column}") DO UPDATE SET {update_sql};'
            )
            
            data_tuples = [tuple(x) for x in df_insert.to_numpy()]
            
            execute_values(cursor, insert_stmt, data_tuples)
            conn.commit()
            logger.info(f"{len(data_tuples)} registros guardados en {qualified_table_name}.")
    except Exception as e:
        logger.error(f"Error al guardar en la base de datos para la tabla '{table_name}': {e}")
        conn.rollback()
    finally:
        if conn: conn.close()
