import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk
from scipy.signal import lombscargle

class ProcesadorECG:
    """
    Clase para el procesamiento modular de señales ECG (pre y post-ejercicio).
    Encapsula el acondicionamiento, filtrado e identificación de QRS.
    """

    def __init__(self, ecg_pre_raw ,ecg_post_raw , fs = 500, resultados_dir="../results/modulo5"):
        self.fs = fs
        self.resultados_dir = resultados_dir
        os.makedirs(self.resultados_dir, exist_ok=True)
        

        # Historial de etapas
        self.senales = {
            'raw': (ecg_pre_raw, ecg_post_raw),
            'raw_recorte':(None, None),
            'hp': (None, None),                       # Pasa-altos (0.67 Hz)
            'bp': (None, None),                       # Pasa-banda Pan-Tompkins
            'der': (None, None),                      # Filtro derivativo
            'sq': (None, None),                       # Elevar al cuadrado
            'mwa': (None, None),                      # Integración móvil
            'candidatos': (None, None),               # Máximos locales de mwa con refractario de 200 ms
            'aceptados':(None, None),                 # Candidatos que el umbral adaptativo clasifica como latidos
            'picos_r': (None, None),                  # Picos R localizados sobre 'hp' desde 'aceptados'
        }

    

    @staticmethod
    def obtener_señal(path_csv):
        """Lee el CSV y devuelve (nombre_columna, valores de la senal)."""
        df = pd.read_csv(path_csv)
        col = df.columns[0]
        return col, df[col].values



    def recorte(self, segundos_recorte):
        recorte = int(segundos_recorte * self.fs) 
        ecg_pre_raw, ecg_post_raw = self.senales['raw']

        ecg_pre_rr = ecg_pre_raw[recorte:]  # rr--> raw recorte
        ecg_post_rr = ecg_post_raw[recorte:]
        self.senales['raw_recorte'] = (ecg_pre_rr, ecg_post_rr)
     
        return ecg_pre_rr, ecg_post_rr


    def graficar_etapa(self, 
                       etapa: str,
                       t_ventana: float, 
                       ecg_pre=None, 
                       ecg_post=None, 
                       titulo_pre=None, 
                       titulo_post=None, 
                       nombre_archivo=None, 
                       mostrar_picos_r=False
                    ):          
            """
            Grafica en dos paneles (Pre y Post-ejercicio) CUALQUIER etapa del pipeline.
            
            Parámetros:
            -----------
            etapa : str
                Nombre de la etapa a graficar ('raw', 'raw_recorte', 'hp', 'bp', 'der', 'sq', 'mwa').
            t_ventana : float
                Segundos de señal a mostrar en el eje horizontal.
            ecg_pre, ecg_post : np.ndarray, opcional
                Si se proveen, grafica estos vectores directamente en lugar de buscarlos en la clase.
            titulo_pre, titulo_post : str, opcional
                Títulos personalizados para los paneles superior e inferior.
            nombre_archivo : str, opcional
                Nombre del archivo PNG para guardar la figura en self.resultados_dir.
            """
            # 1. Si no se pasan vectores externos, se buscan en el diccionario según la 'etapa' ingresada
            if ecg_pre is None or ecg_post is None:
                if etapa not in self.senales:
                    raise KeyError(f"La etapa '{etapa}' no existe. Etapas válidas: {list(self.senales.keys())}")
                
                ecg_pre, ecg_post = self.senales[etapa]

            # 2. Validación de datos procesados
            if ecg_pre is None or ecg_post is None:
                raise ValueError(f"La etapa '{etapa}' aún no ha sido calculada o no contiene datos.")


            # 2b. Validación de picos R (solo si se pidieron)
            r_pre = r_post = None
            if mostrar_picos_r:
                if etapa not in ("raw_recorte", "hp"):
                    raise ValueError(
                        f"Los picos R solo se pueden superponer sobre 'raw_recorte' o 'hp' "
                        f"(los índices se calcularon sobre la señal recortada), no sobre '{etapa}'."
                    )
                r_pre, r_post = self.senales["picos_r"]
                if r_pre is None or r_post is None:
                    raise ValueError("Primero se debe ejecutar localizar_picos_r().")
            # 3. Configuración de estilo visual
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.edgecolor'] = '#cccccc'
            plt.rcParams['axes.linewidth'] = 0.8

            # 4. Construcción del eje de tiempo
            t_pre = np.arange(len(ecg_pre)) / self.fs
            t_post = np.arange(len(ecg_post)) / self.fs

            m_pre = int(t_ventana * self.fs)
            m_post = int(t_ventana * self.fs)

            # 5. Títulos dinámicos según la etapa recibida
            tit_pre = titulo_pre if titulo_pre else f'Pre-ejercicio (Etapa: {etapa.upper()})'
            tit_post = titulo_post if titulo_post else f'Post-ejercicio (Etapa: {etapa.upper()})'

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

            # PANEL 1: PRE-EJERCICIO
            ax1.plot(t_pre[:m_pre], ecg_pre[:m_pre], color='#2b5c8f', linewidth=1.2)
            ax1.set_title(tit_pre, fontsize=12, fontweight='bold', pad=10, loc='left', color='#1a252f')
            ax1.set_ylabel('Amplitud', fontsize=10, color='#2c3e50')
            ax1.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
            ax1.tick_params(colors='#34495e', labelsize=9)
            ax1.set_xlim(0, t_ventana)
            
            if mostrar_picos_r:
                r_visibles = r_pre[r_pre < m_pre]
                ax1.plot(t_pre[r_visibles], ecg_pre[r_visibles], "ro", markersize=5, label="Picos R")
                ax1.legend()

            # PANEL 2: POST-EJERCICIO
            ax2.plot(t_post[:m_post], ecg_post[:m_post], color='#d9534f', linewidth=1.2)
            ax2.set_title(tit_post, fontsize=12, fontweight='bold', pad=10, loc='left', color='#1a252f')
            ax2.set_xlabel('Tiempo [s]', fontsize=10, color='#2c3e50')
            ax2.set_ylabel('Amplitud', fontsize=10, color='#2c3e50')
            ax2.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
            ax2.tick_params(colors='#34495e', labelsize=9)
            ax2.set_xlim(0, t_ventana)
            

            if mostrar_picos_r:
                    r_visibles = r_post[r_post < m_post]
                    ax2.plot(t_post[r_visibles], ecg_post[r_visibles], "ro", markersize=5, label="Picos R")
                    ax2.legend()

            # 6. Guardado en disco en la carpeta configurada en la clase
            if nombre_archivo:
                ruta_guardado = f"{self.resultados_dir}/{nombre_archivo}"
                plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')

            plt.show()
            plt.close(fig) # Libera la memoria de la figura

    def espectro(self, etapa='hp'):
        """
        Grafica la densidad espectral de potencia (Welch) de las señales pre y post
        para decidir si hace falta un notch de 50 Hz.
        """
        pre, post = self.senales[etapa]

        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
        for a, x, titulo in zip(ax, (pre, post), ('Pre-ejercicio', 'Post-ejercicio')):
            f, P = signal.welch(x, fs=self.fs, nperseg=2048)
            a.semilogy(f, P)
            a.axvline(50, color='r', ls='--', label='50 Hz')
            a.set_xlim(0, 100)
            a.set_xlabel('Frecuencia [Hz]')
            a.set_title(titulo)
            a.legend()
        ax[0].set_ylabel('PSD')
        plt.tight_layout()
        plt.show()
        plt.close(fig)



    def filtro_pasa_altos(self, fc=0.67, orden=3):
            """
            Aplica un filtro pasa-altos IIR mediante filtrado forward-backward
            a las señales ECG pre y post-ejercicio. Elimina la deriva de linea de base (las "montañas" por respiración
            o movimiento) dejando pasar las frecuencias mayores a fc.

            Parámetros
            ----------
            fc : float
                Frecuencia de corte [Hz].
            orden : int
                Orden del filtro IIR.

            Guarda
            ------
            self.senales['hp'] : tuple
                Señales pre y post filtradas.
            """
            ecg_pre, ecg_post = self.senales['raw_recorte']

            if ecg_pre is None or ecg_post is None:
                 raise ValueError("Primero se debe ejecutar recorte()")

            # Validar fc
            if not (0 < fc < self.fs / 2):
                 raise ValueError("La frecuencia de corte debe estar entre 0 y fs/2")

            # Filtro IIR Butterworth
            iir = signal.butter(
                 orden,
                 fc,
                 btype="highpass",
                 fs=self.fs,
                 output="sos" # Expresión grande en varias secciones pequeñas de orden máximo 2 ==> confiere estabilidad.
            )

            # Aplicar forward - backward para lograr fase cero.
            ecg_pre_hp = signal.sosfiltfilt(iir, ecg_pre - np.mean(ecg_pre))
            ecg_post_hp = signal.sosfiltfilt(iir, ecg_post - np.mean(ecg_post))

            # Guardar señales filtradas
            self.senales["hp"] = (ecg_pre_hp, ecg_post_hp)

            return ecg_pre_hp, ecg_post_hp
        
    def filtro_pb_pyt(self, fc_baja = 5, fc_alta = 15, orden = 2):
        ecg_pre, ecg_post = self.senales["hp"]

        if ecg_pre is None or ecg_post is None:
            raise ValueError("Ejecutar filtro_pasa_altos() antes.")
    
        if not (0 < fc_baja < fc_alta < self.fs / 2):
         raise ValueError( "Las frecuencias deben cumplir: "
            "0 < frecuencia_baja < frecuencia_alta < fs/2.")

        sos = signal.butter(
        orden,
        [fc_baja, fc_alta],
        btype="bandpass",
        fs=self.fs,
        output="sos"
    )

        # Aplicar el filtro hacia adelante y hacia atrás
        ecg_pre_bp = signal.sosfiltfilt(sos, ecg_pre)
        ecg_post_bp = signal.sosfiltfilt(sos, ecg_post)

        # Guardar las señales filtradas
        self.senales["bp"] = (ecg_pre_bp, ecg_post_bp)

        return ecg_pre_bp, ecg_post_bp

   
    def filtro_derivativo_pyt(self):
        """
        Filtro derivativo de 5 puntos de Pan-Tompkins.

        La salida se expresa en unidades de amplitud por segundo.
        Guarda las derivadas en self.senales['der'].
        """
        bp_pre, bp_post = self.senales["bp"]

        if bp_pre is None or bp_post is None:
            raise ValueError("Primero tenés que ejecutar filtro_pb_pyt().")

        h =( 1/8 * np.array([2, 1, 0, -1, -2]) )*self.fs

        der_pre = np.convolve(bp_pre, h, mode="same")
        der_post = np.convolve(bp_post, h, mode="same")
    
        self.senales["der"] = (der_pre, der_post)

        return der_pre, der_post

    def elevar_al_cuadrado_pyt(self):
        """
        Eleva al cuadrado la señal con filtro derivativo de Pan-Tompkins

        Guarda
        ------
        self.senales['sq'] : tuple
            Señales al cuadrado para los registros pre y post.
        """

        der_pre, der_post = self.senales["der"]

        if der_pre is None or der_post is None:
            raise ValueError("Primero se debe ejecutar filtro_derivativo_pyt().")

        sq_pre = der_pre **2
        sq_post = der_post **2

        self.senales["sq"] = (sq_pre, sq_post)

        return sq_pre, sq_post

    
    def integrar_media_movil_pyt(self, ventana_ms=150):
        """
        Integra la señal elevada al cuadrado mediante una media móvil.

        Parámetros
        ----------
        ventana_ms : float
            Duración de la ventana en milisegundos.

        Guarda
        ------
        self.senales['mwa'] : tuple
            Señales integradas para los registros pre y post.
        """
        sq_pre, sq_post = self.senales["sq"]

        if sq_pre is None or sq_post is None:
            raise ValueError("Primero tenés que ejecutar elevar_al_cuadrado_pyt().")

        # Convertir la duración de la ventana a cantidad de muestras
        N = round(ventana_ms * self.fs / 1000)

        if N < 1:
            raise ValueError("La ventana debe contener al menos una muestra.")

        # Crear la ventana de promedio
        ventana = np.ones(N) / N

        # Aplicar la media móvil
        mwa_pre = np.convolve(sq_pre, ventana, mode="same")
        mwa_post = np.convolve(sq_post, ventana, mode="same")

        self.senales["mwa"] = (mwa_pre, mwa_post)

        return mwa_pre, mwa_post

    
    def detectar_candidatos(self, distancia_minima_ms=200):
        """
        Candidatos a QRS: máximos locales de la señal integrada separados
        al menos distancia_minima_ms (período refractario). Si dos máximos
        están más cerca, se conserva el más alto. Todavía no se distingue
        latido de ruido: eso lo hace umbralizar_adaptativo().

        Guarda
        ------
        self.senales['candidatos'] : tuple
            Índices de los candidatos (pre, post).
        """
        mwa_pre, mwa_post = self.senales["mwa"]

        if mwa_pre is None or mwa_post is None:
            raise ValueError("Primero se debe ejecutar integrar_media_movil_pyt().")

        # Dos QRS no pueden estar más cerca que 200 ms (= 100 muestras a 500 Hz), el período refractario.
        distancia = round(distancia_minima_ms * self.fs / 1000)   # 200 ms = 100 muestras a 500 Hz

        # find_peaks devuelve las posiciones de máximos locales. Con distance = 100, si dos máximos están
        # a menos de 100 muestras, se queda con el más alto.
        cand_pre, _ = signal.find_peaks(mwa_pre, distance=distancia)
        cand_post, _ = signal.find_peaks(mwa_post, distance=distancia)

        self.senales["candidatos"] = (cand_pre, cand_post)
        return cand_pre, cand_post


    def umbralizar_adaptativo(self, n_init_s=2.0, factor_umbral=0.12):
        """
        Clasifica cada candidato como latido o ruido con el umbral adaptativo
        de Pan-Tompkins. SPKI (nivel de señal) y NPKI (nivel de ruido) se
        actualizan con cada candidato, en orden temporal:

            umbral = NPKI + factor_umbral * (SPKI - NPKI)
            si la loma supera el umbral -> latido:  SPKI = 0.125*loma + 0.875*SPKI
            si no                       -> ruido:   NPKI = 0.125*loma + 0.875*NPKI

        SPKI y NPKI arrancan con los primeros n_init_s segundos de la señal.
        El paper original usa 0.25; se bajó a 0.12 para no perder latidos, verificado visualmente.
        Guarda
        ------
        self.senales['aceptados'] : tuple
            Índices de los candidatos clasificados como latido (pre, post).
        """
        mwa_pre, mwa_post = self.senales["mwa"]
        cand_pre, cand_post = self.senales["candidatos"]

        if cand_pre is None or cand_post is None:
            raise ValueError("Primero se debe ejecutar detectar_candidatos().")

        n_init = int(n_init_s * self.fs)

        def clasificar(mwa, candidatos):
            ini = mwa[:n_init]
            spki = 0.5 * np.max(ini)
            npki = 0.5 * np.mean(ini)
            umbral = npki + factor_umbral * (spki - npki)

            aceptados = []
            for p in candidatos:
                if mwa[p] > umbral:
                    spki = 0.125 * mwa[p] + 0.875 * spki # Si supera el umbral, es latido: se guarda y SPKI se corre hacia esa altura.
                    aceptados.append(p)
                else:
                    npki = 0.125 * mwa[p] + 0.875 * npki # De lo contrario, es ruido y NPKI hace lo mismo.
                umbral = npki + factor_umbral * (spki - npki)

            return np.array(aceptados, dtype=int)

        acep_pre = clasificar(mwa_pre, cand_pre)
        acep_post = clasificar(mwa_post, cand_post)

        self.senales["aceptados"] = (acep_pre, acep_post)
        return acep_pre, acep_post

    


    def localizar_picos_r(self, ventana_ms=100):
        """
        Ubica el pico R de cada latido aceptado: la loma de la señal integrada
        es ancha, así que se busca el máximo de la señal con pasa-altos en
        ±ventana_ms alrededor de cada latido.

        Guarda
        ------
        self.senales['picos_r'] : tuple
            Índices de los picos R (pre, post).
        """
        acep_pre, acep_post = self.senales["aceptados"]
        hp_pre, hp_post = self.senales["hp"]

        if acep_pre is None or acep_post is None:
            raise ValueError("Primero se debe ejecutar umbralizar_adaptativo().")

        # Cuánto hacia los costados mira.
        margen = round(ventana_ms * self.fs / 1000)   # 100 ms = 50 muestras a 500 Hz

        def buscar_r(ecg, aceptados):
            picos_r = []
            for a in aceptados:
                inicio = max(0, a - margen)
                final = min(len(ecg), a + margen + 1)
                # argmax da la posición dentro de la ventana; se suma inicio
                # para pasarla a posición en la señal completa
                picos_r.append(inicio + np.argmax(np.abs(ecg[inicio:final])))
            return np.unique(np.array(picos_r, dtype=int))

        r_pre = buscar_r(hp_pre, acep_pre)
        r_post = buscar_r(hp_post, acep_post)

        self.senales["picos_r"] = (r_pre, r_post)
        return r_pre, r_post

    def graficar_deteccion(self, t_ini=0, t_ventana=10, nombre_archivo=None):
        """
        Pasos intermedios de la detección (pre y post) en la ventana [t_ini, t_ini+t_ventana]:
        arriba la señal integrada con candidatos (x gris) y aceptados (círculo verde),
        abajo el ECG con pasa-altos y los picos R finales (rojo).
        """

        a_, b_ = int(t_ini * self.fs), int((t_ini + t_ventana) * self.fs)
        registros = (("Pre-ejercicio", 0, "#2b5c8f"), ("Post-ejercicio", 1, "#d9534f"))

        fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

        for k, (nombre, i, color) in enumerate(registros):
            mwa = self.senales["mwa"][i]
            hp = self.senales["hp"][i]
            t = np.arange(len(hp)) / self.fs

            def en_ventana(idx):
                return idx[(idx >= a_) & (idx < b_)]

            cand = en_ventana(self.senales["candidatos"][i])
            acep = en_ventana(self.senales["aceptados"][i])
            r = en_ventana(self.senales["picos_r"][i])
            desc = np.setdiff1d(cand, acep)

            a1 = ax[2 * k]
            a1.plot(t[a_:b_], mwa[a_:b_], color=color, lw=1.1)
            a1.plot(t[desc], mwa[desc], "x", color="gray", ms=8, label=f"Descartados ({len(desc)})")
            a1.plot(t[acep], mwa[acep], "o", mfc="none", mec="green", ms=10, mew=1.5,
                    label=f"Aceptados ({len(acep)})")
            a1.set_title(f"{nombre}: señal integrada", loc="left", fontweight="bold")

            a2 = ax[2 * k + 1]
            a2.plot(t[a_:b_], hp[a_:b_], color=color, lw=1.1)
            a2.plot(t[r], hp[r], "ro", ms=5, label=f"Picos R ({len(r)})")
            a2.set_title(f"{nombre}: ECG (pasa-altos)", loc="left", fontweight="bold")

            for a in (a1, a2):
                a.grid(True, linestyle="--", alpha=0.5)
                a.set_ylabel("Amplitud")
                a.legend(loc="upper right")

        ax[-1].set_xlabel("Tiempo [s]")
        ax[-1].set_xlim(t_ini, t_ini + t_ventana)
        plt.tight_layout()

        if nombre_archivo:
            plt.savefig(f"{self.resultados_dir}/{nombre_archivo}", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)

class AnalizadorVFC:
    """Analiza y compara la VFC pre y post ejercicio."""

    def __init__(
        self,
        picos_r_pre,
        picos_r_post,
        fs=500,
        resultados_dir="../results/modulo5"
    ):
        self.fs = fs
        self.resultados_dir = resultados_dir
        os.makedirs(self.resultados_dir, exist_ok=True)

        self.picos_r_pre = np.asarray(picos_r_pre, dtype=int)  # Convierte picos R en arrays de NumPy de numeros enteros (en muestras de ECG)
        self.picos_r_post = np.asarray(picos_r_post, dtype=int)

        self.nni_pre = None    
        self.nni_post = None

        self.resultados_tiempo = {}
        self.resultados_frecuencia = {}

    def calcular_rr(self):
        """Calcula los intervalos RR de ambas condiciones."""

        # Calcula las diferencias entre posiciones de picos R.
        # Divide por la frecuencia de muestreo para obtener segundos
        # y multiplica por 1000 para expresar los intervalos en milisegundos.

        self.nni_pre = np.diff(self.picos_r_pre) / self.fs * 1000
        self.nni_post = np.diff(self.picos_r_post) / self.fs * 1000

        # Devuelve los intervalos RR de ambas condiciones.
        return self.nni_pre, self.nni_post

    def graficar_tacogramas(self, nombre_archivo):
        """Grafica y guarda ambos tacogramas en una figura."""

        # Si los intervalos RR todavía no se calcularon, los calcula.
        if self.nni_pre is None:
            self.calcular_rr()

        # Crea una figura con dos gráficos: uno para cada condición.
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))


        # Recorre los datos y el eje correspondiente a cada condición.
        for ax, nni, titulo in [
            (axes[0], self.nni_pre, "Pre ejercicio"),
            (axes[1], self.nni_post, "Post ejercicio")
        ]:
            # Calcula el tiempo acumulado a partir de los intervalos RR.
            tiempo = np.cumsum(nni) / 1000
            ax.plot(tiempo, nni, marker=".", linestyle="-")

            # Configura el título, los ejes y la cuadrícula.
            ax.set_title(titulo)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Intervalo RR (ms)")
            ax.grid(True)

        fig.suptitle("Comparación de tacogramas")
        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            bottom=0.15,
            top=0.82,
            wspace=0.25
)
        # Construye la ruta del archivo utilizando la carpeta de resultados
        # y el nombre indicado al llamar al método.
        ruta = os.path.join(self.resultados_dir,  f"{nombre_archivo}.png")

        # Guarda la figura en formato PNG con resolución de 300 dpi.
        fig.savefig(ruta, dpi=300, bbox_inches="tight")

        # Muestra la figura en el notebook.
        plt.show()

    def calcular_vfc_temporal(self):
        """Calcula las métricas temporales de ambas condiciones."""

        # Calcula los índices temporales a partir de los picos R pre ejercicio.
        self.resultados_tiempo["pre"] = nk.hrv_time(
            peaks=self.picos_r_pre,
            sampling_rate=self.fs,
            show=False
        )


        # Calcula los mismos índices para los picos R post ejercicio.
        self.resultados_tiempo["post"] = nk.hrv_time(
            peaks=self.picos_r_post,
            sampling_rate=self.fs,
            show=False
        )
         # Devuelve un diccionario con los resultados de ambas condiciones.
        return self.resultados_tiempo

    def calcular_vfc_espectral(self):
        """Calcula las métricas espectrales de ambas condiciones."""
        # Calcula las métricas espectrales del registro pre ejercicio.
        self.resultados_frecuencia["pre"] = nk.hrv_frequency(
            peaks=self.picos_r_pre,
            sampling_rate=self.fs,
            show=False
        )

        # Calcula las métricas espectrales del registro post ejercicio.
        self.resultados_frecuencia["post"] = nk.hrv_frequency(
            peaks=self.picos_r_post,
            sampling_rate=self.fs,
            show=False
        )

         # Devuelve los resultados espectrales de ambas condiciones.
        return self.resultados_frecuencia

    
    
    
    def graficar_lomb(self, nombre_archivo="lomb"):
        """Grafica y guarda ambos periodogramas de Lomb-Scargle en una figura."""

          # Define las frecuencias en las que se calculará la potencia espectral.
        frecuencias = np.linspace(0.003, 0.4, 2000)
        potencias = {}

        # Repite el análisis para los registros pre y post ejercicio.
        for picos, etiqueta in [
            (self.picos_r_pre, "Pre ejercicio"),
            (self.picos_r_post, "Post ejercicio")
        ]:
            # Convierte las posiciones de los picos R de muestras a segundos.
            tiempos_r = picos / self.fs
           
            # Calcula los intervalos RR consecutivos en milisegundos.
            nni = np.diff(tiempos_r) * 1000

            # Asocia cada intervalo al instante de su segundo pico R.
            tiempos_rr = tiempos_r[1:]

            # Centra los intervalos restándoles su valor medio.
            rr_centrados = nni - np.mean(nni)

            # Calcula el periodograma de Lomb-Scargle.
            # Las frecuencias se multiplican por 2*pi porque la función
            # recibe frecuencias angulares, expresadas en radianes/segundo.
            potencia = lombscargle(
                tiempos_rr,
                rr_centrados,
                2 * np.pi * frecuencias,
                normalize=True
            )
            # Almacena la potencia asociada a la condición correspondiente.
            potencias[etiqueta] = potencia

        # Una sola figura con dos gráficos
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        for ax, etiqueta in [
            (axes[0], "Pre ejercicio"),
            (axes[1], "Post ejercicio")
        ]:
            ax.plot(frecuencias, potencias[etiqueta])
            ax.set_title(etiqueta)
            ax.set_xlabel("Frecuencia (Hz)")
            ax.set_ylabel("Potencia normalizada")
            ax.grid(True)

        fig.suptitle("Periodogramas de Lomb-Scargle")

        fig.subplots_adjust(
            left=0.08,
            right=0.97,
            bottom=0.15,
            top=0.82,
            wspace=0.25
        )

        ruta = os.path.join(
            self.resultados_dir,
            f"{nombre_archivo}.png"
        )

        fig.savefig(ruta, dpi=300, bbox_inches="tight")
        plt.show()
        # Devuelve las frecuencias y las potencias de ambas condiciones
        # para permitir análisis posteriores sin recalcular el periodograma.
        return frecuencias, potencias["Pre ejercicio"], potencias["Post ejercicio"]

