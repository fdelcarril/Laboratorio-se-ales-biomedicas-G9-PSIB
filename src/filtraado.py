"""
Filtros para limpiar la senal de ECG (AD8232 + Arduino UNO)
Laboratorio de Senales Biomedicas G9 - PSIB
 
Contiene:
  - filtro_pasa_altos: elimina la deriva de linea de base (fc = 0.5 Hz)
  - filtro_notch: elimina el ruido de linea (50 Hz, red electrica)
  - promedio_movil: suaviza el ruido de EMG (actividad muscular, alta frecuencia)
  - resta_linea_base_mediana: saca la deriva residual que deja el pasa-altos
    (complementa al pasa-altos, no lo reemplaza)
"""
 
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
 
FC_PASA_ALTOS = 0.5   # Hz - frecuencia de corte del pasa-altos
NOTCH_FREQ = 50.0     # Hz - frecuencia de la red electrica en Argentina
NOTCH_Q = 30.0        # factor de calidad del notch (mas alto = mas angosto)
VENTANA_MOVIL_MS = 20.0  # ms - ancho de ventana por defecto del promediador movil
VENTANA_MEDIANA_S = 1.0  # s - ancho de ventana por defecto de la mediana movil
 
 
def filtro_pasa_altos(x, fs, fc=FC_PASA_ALTOS, orden=2):
    """Pasa-altos Butterworth de fase cero (filtfilt).
 
    Elimina la deriva de linea de base (las "montanas" por respiracion
    o movimiento) dejando pasar las frecuencias mayores a fc.
    """
    x = np.asarray(x, dtype=float)
    b, a = butter(N=orden, Wn=fc / (fs / 2), btype="highpass")
    return filtfilt(b, a, x)
 
'''
def filtro_notch(x, fs, f0=NOTCH_FREQ, q=NOTCH_Q):
    """Filtro notch IIR de fase cero (filtfilt).
 
    Elimina el ruido de linea (50 Hz) originado por la red electrica
    y los dispositivos usados durante la medicion.
    """
    x = np.asarray(x, dtype=float)
    b, a = iirnotch(w0=f0 / (fs / 2), Q=q)
    return filtfilt(b, a, x)
 
 
def promedio_movil(x, fs, ventana_ms=VENTANA_MOVIL_MS):
    """Promediador movil (filtro FIR de media movil) centrado.
 
    Suaviza el ruido de EMG (actividad muscular, de alta frecuencia)
    promediando cada muestra con sus vecinas dentro de la ventana.
    A mayor ventana, mas suavizado, pero tambien mas se achatan los
    picos R (cuidado con pasarse de ancho).
 
    ventana_ms: ancho de la ventana en milisegundos.
    """
    x = np.asarray(x, dtype=float)
    n = max(1, int(round(ventana_ms / 1000.0 * fs)))
    if n % 2 == 0:
        n += 1  # ventana impar -> centrada exactamente en cada muestra
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")
 
 
def resta_linea_base_mediana(x, fs, ventana_s=VENTANA_MEDIANA_S):
    """Saca la deriva residual estimando la linea de base con una mediana
    movil (ventana larga) y restandola de la senal.
 
    A diferencia del promedio movil, la mediana ignora los picos del QRS
    (son outliers dentro de la ventana), por lo que no los achata: deja
    una linea de base mas plana sin deformar el latido.
 
    ventana_s: ancho de la ventana en segundos (1.0 s por defecto).
    """
    x = np.asarray(x, dtype=float)
    n = max(1, int(round(ventana_s * fs)))
    if n % 2 == 0:
        n += 1
    base = pd.Series(x).rolling(n, center=True, min_periods=1).median().values
    return x - base
'''