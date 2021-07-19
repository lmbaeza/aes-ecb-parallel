#include <stdio.h>

#include<vector>
#include<iostream>
#include<string>
#include<cstdint>
#include<cstring>
#include<cassert>
#include<fstream>

// For the CUDA runtime routines (prefixed with "cuda_")
// #include <cuda_runtime.h>
// #include "/content/aes-ecb-parallel/src/aes.h"
// #include "/content/aes-ecb-parallel/src/file_handler.h"
#include "aes_cuda.h"
#include "file_handler_cuda.h"

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
uint8** text_hex = NULL;

// bloque de 32 posiciones que representa la llave
uint8* key_hex = NULL;

// Mensaje encriptado: la i-th posicion tiene el i-th bloque
uint8** cipher_text = NULL;

// Instancia global del AES
AES aes(256);

int blocks;

void build_hex(string &text, string &key) {
    const int blocks_size = 16;
    int n = (int) text.size();
    blocks = (n+blocks_size-1)/blocks_size;

    text_hex = new uint8*[blocks];
    for(int i = 0; i < blocks; ++i)
        text_hex[i] = new uint8[blocks_size];

    text_to_hex(text, text_hex);

    key_hex = (uint8 *) malloc(32*sizeof(uint8));
    for(int i = 0; i < 32; ++i) key_hex[i] = 0;
    key_to_hex(key, key_hex);
}

void build_ranges(int text_size) {
    intervalo.resize(NUM_HILOS);
    blocks = (text_size+32-1)/32;
    cipher_text = new uint8*[blocks];

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

// int kernel(int ID) {
//     // Rango de bloques a encriptar
//     int from = intervalo[ID].from;
//     int to = intervalo[ID].to;
//     for(int i = from; i <= to; ++i) {
//         uint len = 0;
//         uint8* cipher = aes.EncryptECB(text_hex[i], BLOCK_BYTES_LENGTH, key_hex, len);
//         cipher_text[i] = cipher;
//     }
//     return 0;
// }

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

    string text;
    string key = "admin1234"; // maximo 32 caracteres

    // Leer el texto que se vá a encriptar
    read_file_to_string(NOMBRE_ARCHIVO, text);
    
    build_hex(text, key);

    // Crear los rangos donde van a trabajar lso
    build_ranges((int) text.size());

    /////////////////////////

    for(int i = 0; i < blocks; ++i) {
        uint len = 0;
        uint8* cipher = aes.EncryptECB(text_hex[i], BLOCK_BYTES_LENGTH, key_hex, len);
        cipher_text[i] = cipher;
    }

    write_file(ARCHIVO_SALIDA, cipher_text, blocks, 16);

    return 0;
}