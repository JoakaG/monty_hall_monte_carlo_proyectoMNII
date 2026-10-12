import os
import re
import csv
import math
import subprocess
import statistics

def compilar_codigo_c(fuente_c="monty_hall_general_mpi_2.c", ejecutable="monty_hall_c"):
    if not os.path.exists(fuente_c) and os.path.exists("monty_hall_general_mpi.c"):
        fuente_c = "monty_hall_general_mpi.c"

    print(f"Compilando '{fuente_c}' con mpicc -O3 -lm...", flush=True)
    cmd = ["mpicc", "-O3", fuente_c, "-o", ejecutable, "-lm"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error de compilación:\n{res.stderr}", flush=True)
        return False
    print("Compilación exitosa.\n", flush=True)
    return True

def parsear_salida_c(salida_texto):
    match_tiempo = re.search(r"Tiempo de ejecución.*?:\s+([\d\.]+)\s*s", salida_texto, re.IGNORECASE)
    match_mantiene = re.search(r"(?:Frecuencia empírica|Ganar MANTENIENDO).*?:\s+(?:[\d]+\s*\()?([\d\.]+)%", salida_texto, re.IGNORECASE)
    
    tiempo = float(match_tiempo.group(1)) if match_tiempo else None
    pct_mantiene = float(match_mantiene.group(1)) if match_mantiene else None

    return tiempo, pct_mantiene

def ejecutar_benchmark_cluster():
    ejecutable = "./monty_hall_c"
    archivo_csv = "resultados_monty_hall.csv"

    if not compilar_codigo_c():
        return

    # Grilla de simulaciones y muestreo fino entre 64 y 96 cores
    simulaciones_lista = [100, 10000, 1000000, 10000000, 1000000000, 10000000000] # N = 10^6, 10^7, 10^9, 10^10
    puertas_lista = [3, 10, 100]
    
    # Muestreo detallado entre 64 y 96 para analizar la degradación NUMA/Infinity Fabric
    procesos_lista = [1, 2, 4, 8, 16, 32, 48, 64, 72, 80, 88, 96]

    MAX_TIMEOUTS_PERMITIDOS = 1  # Cancelación rápida si ocurren 2 timeouts seguidos

    headers = [
        'puertas', 
        'simulaciones_totales', 
        'procesadores', 
        'tiempo_mediana_segundos', 
        'speedup', 
        'eficiencia_porcentaje', 
        'pct_gana_mantiene', 
        'pct_gana_cambia', 
        'error_absoluto_pct', 
        'error_relativo_pct', 
        'error_estocastico_se', 
        'repeticiones'
    ]

    with open(archivo_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print("=== INICIANDO BENCHMARK OPTIMIZADO (CON CIRCUIT BREAKER Y MUESTREO FINO) ===", flush=True)
    print(f"Resultados guardados en: {archivo_csv}\n", flush=True)

    total_combinaciones = len(puertas_lista) * len(simulaciones_lista)
    contador = 0

    for K in puertas_lista:
        p_teorico_mantiene = (1.0 / K) * 100.0

        for N in simulaciones_lista:
            contador += 1
            print(f"\n[{contador}/{total_combinaciones}] Evaluando K = {K} puertas | N = {N:,} simulaciones", flush=True)

            t1_mediana = None
            repeticiones = 5 if N >= 1000000000 else 10

            p_teor_prop = 1.0 / K
            se_monte_carlo = math.sqrt((p_teor_prop * (1.0 - p_teor_prop)) / N) * 100.0

            for p in procesos_lista:
                tiempos_obtenidos = []
                mantiene_pcts = []
                timeouts_consecutivos = 0

                # Timeout dinámico: da más margen a p=1 para N=10^10 (evita perder T1)
                if N == 10000000000 and p == 1:
                    timeout_limite = 300  # 5 minutos para p=1 secuencial masivo
                elif N >= 1000000000:
                    timeout_limite = 120
                else:
                    timeout_limite = 60

                for r in range(1, repeticiones + 1):
                    # Si ya falló 2 veces seguidas, interrumpe esta configuración
                    if timeouts_consecutivos >= MAX_TIMEOUTS_PERMITIDOS:
                        print(f"   [CIRCUIT BREAKER] Se alcanzaron {MAX_TIMEOUTS_PERMITIDOS} timeouts seguidos para p={p}. Cancelando repeticiones y pasando al siguiente número de procesadores.", flush=True)
                        break

                    cmd = ["mpirun", "-n", str(p), ejecutable, str(N), str(K)]

                    try:
                        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_limite)
                        if res.returncode != 0: 
                            continue

                        t, pm = parsear_salida_c(res.stdout)
                        if t is not None:
                            tiempos_obtenidos.append(t)
                            if pm is not None: mantiene_pcts.append(pm)
                            timeouts_consecutivos = 0  # Reiniciar contador de timeouts si tuvo éxito
                    except subprocess.TimeoutExpired:
                        timeouts_consecutivos += 1
                        print(f"   [TIMEOUT {timeouts_consecutivos}/{MAX_TIMEOUTS_PERMITIDOS}] Corrida {r} excedió {timeout_limite}s para p={p}, N={N:,}, K={K}", flush=True)
                        continue

                if tiempos_obtenidos:
                    mediana_t = statistics.median(tiempos_obtenidos)
                    prom_mantiene = statistics.mean(mantiene_pcts) if mantiene_pcts else p_teorico_mantiene
                    prom_cambia = 100.0 - prom_mantiene

                    if p == 1:
                        t1_mediana = mediana_t
                        speedup = 1.0000
                        eficiencia = 100.00
                    else:
                        if t1_mediana is not None and mediana_t > 0:
                            speedup = t1_mediana / mediana_t
                            eficiencia = (speedup / p) * 100.0
                        else:
                            speedup, eficiencia = 0.0, 0.0

                    err_abs = abs(p_teorico_mantiene - prom_mantiene)
                    err_rel = (err_abs / p_teorico_mantiene) * 100.0 if p_teorico_mantiene > 0 else 0.0

                    with open(archivo_csv, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            K, N, p, 
                            f"{mediana_t:.6f}", 
                            f"{speedup:.4f}", 
                            f"{eficiencia:.2f}", 
                            f"{prom_mantiene:.6f}", 
                            f"{prom_cambia:.6f}", 
                            f"{err_abs:.6f}", 
                            f"{err_rel:.6f}", 
                            f"{se_monte_carlo:.8f}", 
                            len(tiempos_obtenidos)
                        ])

                    print(f"   p:{p:2d} cores | T_p:{mediana_t:.6f}s | Sp:{speedup:.2f}x | Ep:{eficiencia:.1f}% | ErrRel:{err_rel:.4f}%", flush=True)
                else:
                    print(f"   p:{p:2d} cores | OMITIDO por timeouts/fallas.", flush=True)

    print(f"\n¡Benchmark optimizado completado! Datos consolidados en '{archivo_csv}'.", flush=True)

if __name__ == "__main__":
    ejecutar_benchmark_cluster()