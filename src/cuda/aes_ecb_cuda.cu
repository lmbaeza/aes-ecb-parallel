%%writefile cuda_sample.cu
#include <stdio.h>

#include<vector>
#include<iostream>
#include<string>
#include<cstdint>
#include<cstring>
#include<cassert>
#include<fstream>

#include <cuda_runtime.h>

#include "aes_cuda.h"
#include "file_handler_cuda.h"

using namespace std;

string NOMBRE_ARCHIVO;
string ARCHIVO_SALIDA;

int NUM_HILOS;

int *pArgc = NULL;

char** pArgv = NULL;

// el i-th hilo procesa el intervalo, [from, to]
int (*intervalo)[2];

// texto dividido en bloques: la i-th posicion tiene el i-th bloque
uint8 (*text_hex)[BLOCKS_SIZE] = NULL;

// bloque de 32 posiciones que representa la llave
uint8* key_hex = NULL;

// Mensaje encriptado: la i-th posicion tiene el i-th bloque
uint8 (*cipher_text)[BLOCKS_SIZE] = NULL;

// Instancia global del AES

int blocks;

// __device__
// int d_blocks;

void build_hex(string &text, string &key) {
    int n = (int) text.size();
    blocks = (n+BLOCKS_SIZE-1)/BLOCKS_SIZE;

    text_hex = new uint8[blocks][BLOCKS_SIZE];
    // for(int i = 0; i < blocks; ++i)
    //     text_hex[i] = new uint8[BLOCKS_SIZE];

    text_to_hex(text, text_hex);

    key_hex = (uint8 *) malloc(32*sizeof(uint8));
    for(int i = 0; i < 32; ++i) key_hex[i] = 0;
    key_to_hex(key, key_hex);
}

void build_ranges(int text_size) {
    intervalo = new int[NUM_HILOS][2];
    for(int i = 0; i < NUM_HILOS; ++i) {
        // intervalo[i] = new int[2];
        intervalo[i][0] = 0;
        intervalo[i][1] = 0;
    }
    
    // for(int i = 0; i < NUM_HILOS; ++i) {
    //     intervalo[i] = (int *) malloc(2*sizeof(int));
    //     intervalo[i][0] = 0;
    //     intervalo[i][1] = 0;
    // }

    blocks = (text_size+32-1)/32;
    
    cipher_text = new uint8[blocks][BLOCKS_SIZE];

    // Crear Intervalos
    int len = blocks/NUM_HILOS;
    vector<int> length(NUM_HILOS);
    for(int i = 0; i < NUM_HILOS; ++i) length[i] = len;
    int total = blocks - len*NUM_HILOS;
    for(int i = 0; i < total; ++i) length[i]++;
    int current = 0;
    for(int i = 0; i < NUM_HILOS; ++i) {
        intervalo[i][0] = current;
        intervalo[i][1] = current+length[i]-1;
        current = current + length[i];
    }
    intervalo[NUM_HILOS-1][1] = blocks-1;
}


__global__ 
void kernel(int (*k_intervalo)[2], uint8 (*k_cipher_text)[BLOCKS_SIZE], uint8 (*k_text_hex)[BLOCKS_SIZE], uint8* k_key_hex, AES* aes, int * d_blocks) {
    int ID = blockIdx.x * blockDim.x + threadIdx.x;

    printf("\nGPU ID=%d blocks=%d", ID, (*d_blocks));

    if (ID < (*d_blocks)) {

        printf(" -- Inside ID=%d\n", ID);
        int from = k_intervalo[ID][0];
        int to = k_intervalo[ID][1];
        
        printf("\nGPU from=%d to=%d\n", from, to);

        for(int i = from; i <= to; ++i) {
            printf("GPU #1\n");
            uint len = 0;
            printf("GPU #2\n");
            uint8* cipher = aes->EncryptECB(k_text_hex[i], 16 * sizeof(uint8), k_key_hex, len);
            printf("GPU #3\n");
            for(int j = 0; j < BLOCKS_SIZE; ++j) {
                k_cipher_text[i][j] = cipher[j];
            }
            //k_cipher_text[i] = cipher;

            printf("GPU: ");
            for(int j = 0; j < 16; ++j) {
                uint8 val = k_cipher_text[i][j];
                printf("%02x ", val);
            }
            printf("\n");
        }
        // k_cipher_text vá a ser el unico modificado
    }
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
 
    printf("%s %s %d\n", NOMBRE_ARCHIVO.c_str(), ARCHIVO_SALIDA.c_str(), NUM_HILOS);

    string text;
    string key = "admin1234"; // maximo 32 caracteres

    // Leer el texto que se vá a encriptar
    read_file_to_string(NOMBRE_ARCHIVO, text);

    text = "12345343243534534564365656546546";
    printf("Text: [%s]\n", text.c_str());
    
    build_hex(text, key);

    // Crear los rangos donde van a trabajar lso
    build_ranges((int) text.size());


    int (*d_intervalo)[2];
    cudaMalloc(&d_intervalo, NUM_HILOS*2*sizeof(int));
    cudaMemcpy(d_intervalo, intervalo, NUM_HILOS*2*sizeof(int), cudaMemcpyHostToDevice);

    // cipher_text
    uint8 (*d_cipher_text)[BLOCKS_SIZE];
    cudaMalloc(&d_cipher_text, blocks*BLOCKS_SIZE*sizeof(uint8));
    cudaMemcpy(d_cipher_text, cipher_text,  blocks*BLOCKS_SIZE*sizeof(uint8), cudaMemcpyHostToDevice);

    // text_hex
    uint8 (*d_text_hex)[BLOCKS_SIZE];
    // float (*d_C)[N];
    cudaMalloc((void**)(&d_text_hex), blocks*BLOCKS_SIZE*sizeof(uint8));

    cudaMemcpy(d_text_hex, text_hex,  blocks*BLOCKS_SIZE*sizeof(uint8), cudaMemcpyHostToDevice);

    // key_hex
    uint8* d_key_hex = NULL;
    cudaMalloc(&d_key_hex, 32*sizeof(uint8));
    cudaMemcpy(d_key_hex, key_hex,  32*sizeof(uint8), cudaMemcpyHostToDevice);

    AES aes(256);
    AES *d_aes;
    cudaMalloc(&d_aes, sizeof(AES));
    cudaMemcpy(d_aes, &aes,  sizeof(AES), cudaMemcpyHostToDevice);

    printf("CPU blocks=%d\n", blocks);

    int *d_blocks;
    cudaMalloc(&d_blocks, sizeof(int));

    cudaMemcpy(d_blocks, &blocks,  sizeof(int), cudaMemcpyHostToDevice);
    
    kernel<<<1, 1>>>(d_intervalo, d_cipher_text, d_text_hex, d_key_hex, d_aes, d_blocks);
    cudaDeviceSynchronize();

    cudaMemcpy(cipher_text, d_cipher_text, blocks*BLOCKS_SIZE*sizeof(uint8), cudaMemcpyDeviceToHost);

    // write_file(ARCHIVO_SALIDA, cipher_text, blocks, 16);

    printf("###################################\n");

    for(int i = 0; i < blocks; ++i) {
        for(int j = 0; j < 16; ++j) {
            uint8 val = cipher_text[i][j];
            if(val < 16) cout << '0';
            cout << hex << (int) val;
        }
        cout << '\n';
    }

    return 0;
}

// 78cdb9aa782851e8502e5d8da6927b25