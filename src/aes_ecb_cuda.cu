%%writefile cuda_sample.cu

#include <stdio.h>

#include<vector>
#include<iostream>
#include<string>
#include<cstdint>
#include<cstring>
#include<cassert>
#include<fstream>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include "/content/aes-ecb-parallel/src/aes.h"
#include "/content/aes-ecb-parallel/src/file_handler.h"
//#include "aes.h"
//#include "file_handler.h"

using namespace std;

string NOMBRE_ARCHIVO;
string ARCHIVO_SALIDA;
int NUM_HILOS;

int *pArgc = NULL;
char** pArgv = NULL;

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

}

int main(int argc, char **argv) {   
    pArgc = &argc;
    pArgv = argv;
 
    if((*pArgc) < 4) {
        printf("Debe proporcionar 3 argumentos: [archivo de entrada] [archivo de salida] [numero de hilos]");
        // Ejemplo: ./filtro.o img/input1.png img/output1.png 8 16
        exit(0);
    }
    // Ruta del archivo de entrada: Ej: input1.txt
    NOMBRE_ARCHIVO = string(pArgv[1]);

    // Ruta del texto cifrado: Ej: output.bin
    ARCHIVO_SALIDA = string(pArgv[2]);

    // Numero de hilos utilizados
    NUM_HILOS = atoi(pArgv[3]);
 
    printf("%s %s %d", NOMBRE_ARCHIVO.c_str(), ARCHIVO_SALIDA.c_str(), NUM_HILOS);

    return 0;
}