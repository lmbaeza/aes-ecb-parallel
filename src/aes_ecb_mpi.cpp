// mpiCC -lm -Ofast -std=c++17 -o aes_ecb_mpi src/aes_ecb_mpi.cpp && time mpirun -np 2 aes_ecb_mpi input.txt output1.bin

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <unistd.h>
#include <sys/time.h>
#include <assert.h>

#include <mpi.h>

// Include Aes Algorithm
#include "aes.h"
#include "file_handler.h"

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

#define unreachable assert(false && "Unreachable");

// Instancia global del AES
AES aes(256);

void build_hex(string &text, string &key) {
    text_hex = text_to_hex(text);
    key_hex = key_to_hex(key);
}

void build_ranges(int text_size) {
    intervalo.resize(NUM_HILOS);
    int blocks = (text_size+16-1)/16;
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

char* substr_text(const string &text, int from, int to) {
    int len = to - from + 1;
    // int size = (len+32-1)/32;
    char* output = (char *) malloc(16 * len * sizeof(char));
    int idx = 0;
    for(int i = from; i <= to; ++i) {
        vector<uint8> current = text_hex[i];
        for(int j = 0; j < 16; ++j) output[idx++]= (char) current[j];
    }
    return output;
}

int main(int argc, char *argv[]) {

    if(argc < 3) {
        cout << "Debe proporcionar 3 argumentos: [archivo de entrada] [archivo de salida] [numero de hilos]" << endl;
        // Ejemplo: ./filtro.o img/input1.png img/output1.png 8 16
        exit(0);
    }

    int tasks, current_id, root = 0, tag = 1;
    MPI_Status status;

    #define is_root(id) (root == id)
    #define is_node(id) (root != id)

    MPI_Init(&argc , &argv);
    MPI_Comm_size( MPI_COMM_WORLD , &tasks);
    MPI_Comm_rank( MPI_COMM_WORLD , &current_id);

    int from, to, len;
    char* substring = NULL;

    string text;
    string key = "admin1234"; // maximo 32 caracteres
    key_hex = key_to_hex(key);

    vector<vector<char>> cipher_by_tasks(tasks);

    if(is_root(current_id)) {
        // Ruta del archivo de entrada: Ej: input1.txt
        NOMBRE_ARCHIVO = string(argv[1]);

        // Ruta del texto cifrado: Ej: output.bin
        ARCHIVO_SALIDA = string(argv[2]);

        NUM_HILOS = tasks;

        // Leer el texto que se vá a encriptar
        read_file_to_string(NOMBRE_ARCHIVO, text);

        build_hex(text, key);

        // Crear los rangos donde van a trabajar lso
        build_ranges((int) text.size());
    }

    if(is_root(current_id)) {
        for(int i = 0; i < tasks; ++i) {
            if(i == root)
                continue;
            // Rango de bloques a encriptar
            from = intervalo[i].from;
            to = intervalo[i].to;
            len = to - from + 1;

            int limits[3] = {from, to, len};
            // Enviar Limites
            MPI_Send(&(limits[0]), 3, MPI_INT, i, tag, MPI_COMM_WORLD);

            substring = substr_text(text, from, to);
            // Enviar texto a encriptar
            MPI_Send(&substring[0], 16*len, MPI_CHAR, i, tag, MPI_COMM_WORLD);
        }

        from = intervalo[root].from;
        to = intervalo[root].to;
        len = to - from + 1;
    }
    
    if(is_root(current_id)) {
        // guardar los datos de substring, para ser procesados
        substring = substr_text(text, from, to);
    } else if(is_node(current_id)) {
        // Recibir los Limites
        int limits[3];
        MPI_Recv(&(limits[0]), 3, MPI_INT, root, tag, MPI_COMM_WORLD, &status);
        from = limits[0];
        to = limits[1];
        len = limits[2];

        if(substring != NULL)
            free(substring);
        
        substring = (char*) malloc(16 * len * sizeof(char));

        // Recibir texto a encriptar
        MPI_Recv(&substring[0], 16*len, MPI_CHAR, root, tag, MPI_COMM_WORLD, &status);

    } else {
        unreachable;
    }

    // Encriptar

    char* output = (char*) malloc(16 * len * sizeof(char));

    int idx = 0;
    int idx_out = 0;
    vector<uint8> block(16, 0);

    for(int i = 0; i < len; ++i) {
        uint tmp_len = 0;
        for(int j = 0; j < 16; ++j) block[j] = substring[idx++];

        vector<uint8> cipher = aes.EncryptECB(block, BLOCK_BYTES_LENGTH, key_hex, tmp_len);
        for(int j = 0; j < 16; ++j) output[idx_out++] = 0;//(char) cipher[j];
    }

    if(is_root(current_id)) {
        cipher_by_tasks[root].reserve(16*len);
        for(int j = 0; j < 16*len; ++j)
            cipher_by_tasks[root].push_back(substring[j]);
    } else if(is_node(current_id)) {
        // Enviar texto Encriptado
        MPI_Send(&output[0], 16*len, MPI_CHAR, root, tag, MPI_COMM_WORLD);
    } else {
        unreachable;
    }

    if(is_root(current_id)) {
        for(int i = 0; i < tasks; ++i) {
            if(i == root)
                continue;
            // Recibir datos encriptados
            from = intervalo[i].from;
            to = intervalo[i].to;
            len = to - from + 1;

            substring = (char *) malloc(16 * len * sizeof(char));
            MPI_Recv(&substring[0], 16*len, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);

            cipher_by_tasks[i].reserve(16*len);

            for(int j = 0; j < 16*len; ++j)
                cipher_by_tasks[i].push_back(substring[j]);
        }
    }

    if(is_root(current_id)) {
        int x = 0;
        int y = 0;
        for(int i = 0; i < (int) cipher_text.size(); ++i) {
            cipher_text[i].resize(16);
        }

        for(int i = 0; i < tasks; ++i) {
            for(int j = 0; j < (int) cipher_by_tasks[i].size(); ++j) {
                cipher_text[x][y] = (uint8) cipher_by_tasks[i][j];
                y = (y+1) % 16;
                if(y == 0)
                    x++;
            }
        }
        // Guardar el cifrado en un archivo .bin
        write_file(ARCHIVO_SALIDA, cipher_text);
    }

    printf("Process %d of %d\n", current_id, tasks);

    free(substring);

    MPI_Finalize();
    
    return 0;
}