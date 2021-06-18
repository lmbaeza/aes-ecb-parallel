#include <bits/stdc++.h>
#include<vector>
#include<iostream>
#include<string>
#include<cstdint>
#include<cstring>
#include<cassert>

// Headers for Posix
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

// Include Aes Algorithm
#include "aes.h"
#include "file_handler.h"

using namespace std;

string NOMBRE_ARCHIVO;
string ARCHIVO_SALIDA;
int NUM_HILOS;

struct Rango {
    int from;
    int to;
};
// el i-th hilo procesa el intervalo, [from, to]
vector<Rango> intervalo;

// texto dividido en bloques: la i-th posicion tiene el i-th bloque
vector<vector<uint8>> text_hex;

// bloque de 32 posiciones que representa la llave
vector<uint8> key_hex;

// Mensaje encriptado: la i-th posicion tiene el i-th bloque
vector<vector<uint8>> cipher_text;

// Instancia global del AES
AES aes(256);

void build_hex(string &text, string &key) {
    text_hex = text_to_hex(text);
    key_hex = key_to_hex(key);
}

void build_ranges(int text_size) {
    intervalo.resize(NUM_HILOS);
    int blocks = (text_size+32-1)/32;
    cipher_text.resize(blocks);
    // Crear Intervalos
    int len = blocks/NUM_HILOS;
    vector<int> length(NUM_HILOS);
    for(int i = 0; i < NUM_HILOS; ++i) length[i] = len;
    int total = blocks - len*NUM_HILOS;
    for(int i = 0; i < total; ++i) length[i]++;
    int current = 0;
    for(int i = 0; i < NUM_HILOS; ++i) {
        intervalo[i] = Rango {current, current+length[i]-1};
        current = current + length[i];
    }
    intervalo[NUM_HILOS-1].to = blocks-1;

    //for(int i = 0; i < NUM_HILOS; ++i) cout << intervalo[i].from << " " << intervalo[i].to << endl;
}

void * run_parallel_encryption_posix(void *arg) {
    // Id del Hilo
    int threadId = *(int*) arg;

    // Rango de bloques a encriptar
    int from = intervalo[threadId].from;
    int to = intervalo[threadId].to;

    for(int i = from; i <= to; ++i) {
        uint len = 0;
        vector<uint8> cipher = aes.EncryptECB(text_hex[i], BLOCK_BYTES_LENGTH, key_hex, len);
        cipher_text[i] = cipher;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    if(argc < 4) {
        cout << "Debe proporcionar 3 argumentos: [archivo de entrada] [archivo de salida] [numero de hilos]" << endl;
        // Ejemplo: ./filtro.o img/input1.png img/output1.png 8 16
        exit(0);
    }
    // Ruta del archivo de entrada: Ej: input1.txt
    NOMBRE_ARCHIVO = string(argv[1]);

    // Ruta del texto cifrado: Ej: output.bin
    ARCHIVO_SALIDA = string(argv[2]);

    // Numero de hilos utilizados
    NUM_HILOS = atoi(argv[3]);

    string text;
    string key = "admin1234"; // maximo 32 caracteres

    // Leer el texto que se vá a encriptar
    read_file_to_string(NOMBRE_ARCHIVO, text);

    build_hex(text, key);

    // Crear los rangos donde van a trabajar lso
    build_ranges((int) text.size());

    // Crear los Hilos
    int threadId[NUM_HILOS];
    pthread_t thread[NUM_HILOS];

    // Definir variables para medir el tiempo de ejecucion
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    
    // Crear los hilos
    for(int i = 0; i < NUM_HILOS; i++){
        threadId[i] = i;
        pthread_create(&thread[i], NULL, run_parallel_encryption_posix, &threadId[i]);
    }

    // Unir los hilos
    int *retval;
    for(int i = 0; i < NUM_HILOS; i++){
        pthread_join(thread[i], (void **)&retval);
    }

    // Medir el tiempo
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);

    // Mostrar el tiempo de ejecución
    printf("Time elapsed: %ld.%06ld using %d threads\n", (long int) tval_result.tv_sec, (long int) tval_result.tv_usec, NUM_HILOS);

    // Guardar el cifrado en un archivo .bin
    write_file(ARCHIVO_SALIDA, cipher_text);
    
    return 0;
}
