import matplotlib.pyplot as plt
import numpy as np

def graficar_ecg_comparativo(ecg_pre, ecg_post, fs, titulo_pre, titulo_post, ruta_guardado, t_ventana):
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8

    t_pre = np.arange(len(ecg_pre)) / fs
    t_post = np.arange(len(ecg_post)) / fs

    m_pre = int(t_ventana * fs)
    m_post = int(t_ventana * fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=False)

    # PANEL 1: PRE-EJERCICIO
    ax1.plot(t_pre[:m_pre], ecg_pre[:m_pre], color='#2b5c8f', linewidth=1.2)
    ax1.set_title(titulo_pre, fontsize=12, fontweight='bold', pad=10, loc='left', color='#1a252f')
    ax1.set_ylabel('Amplitud', fontsize=10, color='#2c3e50')
    ax1.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
    ax1.tick_params(colors='#34495e', labelsize=9)
    ax1.set_xlim(0, t_ventana)

    # PANEL 2: POST-EJERCICIO
    ax2.plot(t_post[:m_post], ecg_post[:m_post], color='#d9534f', linewidth=1.2)
    ax2.set_title(titulo_post, fontsize=12, fontweight='bold', pad=10, loc='left', color='#1a252f')
    ax2.set_xlabel('Tiempo [s]', fontsize=10, color='#2c3e50')
    ax2.set_ylabel('Amplitud', fontsize=10, color='#2c3e50')
    ax2.grid(True, linestyle='--', alpha=0.5, color='#bdc3c7')
    ax2.tick_params(colors='#34495e', labelsize=9)
    ax2.set_xlim(0, t_ventana)

    plt.tight_layout()
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig) # Libera memoria