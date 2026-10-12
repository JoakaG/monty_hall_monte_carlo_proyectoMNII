#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

/**
 * Simulación Generalizada de Monty Hall para K puertas con C + MPI y Análisis de Errores.
 * 
 * @param total_simulaciones Número total de iteraciones masivas N.
 * @param num_puertas Número de puertas en juego K (K >= 3).
 */
void simular_monty_hall_general(long long total_simulaciones, int num_puertas) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Validación única de parámetros
    if (num_puertas < 3) {
        if (rank == 0) {
            printf("Error: El número de puertas K debe ser al menos 3.\n");
        }
        return;
    }

    // Sincronización inicial y toma de tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Reparto equitativo del tamaño de bloque (chunk) manejando el residuo en el último proceso
    long long chunk_size = total_simulaciones / size;
    if (rank == size - 1) { 
        chunk_size += total_simulaciones % size; 
    }

    // Semilla aleatoria única e independiente por cada rank MPI
    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)((rank + 1) * 7919);

    long long mantiene_gana_local = 0;

    // Bucle de simulación Monte Carlo de alto rendimiento (sin condicionales complejos)
    for (long long i = 0; i < chunk_size; i++) {
        int premio = rand_r(&seed) % num_puertas;
        int eleccion = rand_r(&seed) % num_puertas;

        // Si la elección inicial es igual al premio, la estrategia de Mantener gana
        if (premio == eleccion) {
            mantiene_gana_local++;
        }
    }

    // Reducción colectiva de suma hacia el proceso raíz (rank 0)
    long long mantiene_gana_global = 0;
    MPI_Reduce(&mantiene_gana_local, &mantiene_gana_global, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Medición final de tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    double tf = MPI_Wtime();

    // El proceso raíz procesa los resultados, métricas y cálculo de errores
    if (rank == 0) {
        double tiempo_ejecucion = tf - t0;
        long long cambia_gana_global = total_simulaciones - mantiene_gana_global;

        // Porcentajes empíricos obtenidos
        double pct_mantiene = ((double)mantiene_gana_global / (double)total_simulaciones) * 100.0;
        double pct_cambia = ((double)cambia_gana_global / (double)total_simulaciones) * 100.0;

        // Valores teóricos exactos
        double teorico_mantiene = (1.0 / (double)num_puertas) * 100.0;
        double teorico_cambia = (((double)num_puertas - 1.0) / (double)num_puertas) * 100.0;

        // 1. Error Absoluto Verdadero (|Et|)
        double err_abs_mantiene = fabs(teorico_mantiene - pct_mantiene);
        double err_abs_cambia = fabs(teorico_cambia - pct_cambia);

        // 2. Error Relativo Porcentual Verdadero (e_t %)
        double err_rel_pct_mantiene = (err_abs_mantiene / teorico_mantiene) * 100.0;
        double err_rel_pct_cambia = (err_abs_cambia / teorico_cambia) * 100.0;

        // 3. Error Estándar Teórico de Monte Carlo (SE = sqrt(p*(1-p)/N) * 100)
        double p_teorico = 1.0 / (double)num_puertas;
        double se_monte_carlo = sqrt((p_teorico * (1.0 - p_teorico)) / (double)total_simulaciones) * 100.0;

        // Reporte consolidado de ejecución y análisis numérico
        printf("\n=====================================================================\n");
        printf("===     SIMULACIÓN MONTY HALL GENERALIZADA (C + MPI + ERRORES)    ===\n");
        printf("=====================================================================\n");
        printf("Puertas en juego (K):           %d\n", num_puertas);
        printf("Procesos MPI (p):               %d\n", size);
        printf("Simulaciones totales (N):       %lld\n", total_simulaciones);
        printf("Tiempo de ejecución (T_p):      %.6f s\n", tiempo_ejecucion);
        printf("---------------------------------------------------------------------\n");
        printf("RESULTADOS ESTRATEGIA: MANTENIENDO LA PUERTA\n");
        printf("  - Victorias obtenidas:        %lld\n", mantiene_gana_global);
        printf("  - Frecuencia empírica:        %.6f%%\n", pct_mantiene);
        printf("  - Probabilidad teórica:       %.6f%%\n", teorico_mantiene);
        printf("  - Error Absoluto Verdadero:   %.6f%%\n", err_abs_mantiene);
        printf("  - Error Relativo Porcentual:  %.6f%%\n", err_rel_pct_mantiene);
        printf("---------------------------------------------------------------------\n");
        printf("RESULTADOS ESTRATEGIA: CAMBIANDO LA PUERTA\n");
        printf("  - Victorias obtenidas:        %lld\n", cambia_gana_global);
        printf("  - Frecuencia empírica:        %.6f%%\n", pct_cambia);
        printf("  - Probabilidad teórica:       %.6f%%\n", teorico_cambia);
        printf("  - Error Absoluto Verdadero:   %.6f%%\n", err_abs_cambia);
        printf("  - Error Relativo Porcentual:  %.6f%%\n", err_rel_pct_cambia);
        printf("---------------------------------------------------------------------\n");
        printf("ERROR ESTÁNDAR DE MONTE CARLO (SE): %.8f%%\n", se_monte_carlo);
        printf("=====================================================================\n\n");
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // Parámetros por defecto
    long long total_simulaciones = 100000000LL; // 100M
    int num_puertas = 3;                       // K = 3 por defecto

    // Lectura de parámetros desde línea de comandos: ./monty_hall_c [N] [K]
    if (argc > 1) {
        total_simulaciones = atoll(argv[1]);
    }
    if (argc > 2) {
        num_puertas = atoi(argv[2]);
    }

    simular_monty_hall_general(total_simulaciones, num_puertas);

    MPI_Finalize();
    return 0;
}