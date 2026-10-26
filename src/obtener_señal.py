"""
Agarra la señal de ECG desde un archivo CSV (una columna de muestras).
"""
 
import pandas as pd
 
 
def agarrar_senal(path_csv):
    """Lee el CSV y devuelve (nombre_columna, valores de la senal)."""
    df = pd.read_csv(path_csv)
    col = df.columns[0]
    return col, df[col].values
 