#include<iostream>
#include<string>
#include<vector>

using namespace std;

#include "file_handler.h"

const string kilobyte = "What is Lorem Ipsum?\nLorem Ipsum is simply dummy text of the printing and- typesetting industry.\nLorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.\nIt has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged.\nIt was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.\nWhy do we use it? It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout.\nThe point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English.\nMany desktop publishing packages and web page editors now use Lorem Ipsum as ..\n";

void generate(int megabytes, const string &filename) {
    // 1 kilobyte = 1000 bytes
    string accumulate = "";
    accumulate.reserve(1'000'000*megabytes);

    for(int i = 0; i < megabytes*1000; ++i) {
        accumulate += kilobyte;
    }

    write_file_string(filename, accumulate);
}

int main(int argc, char *argv[]) {
    // Ruta del archivo de entrada: Ej: input1.txt
    string NOMBRE_ARCHIVO = string(argv[1]);
    // Generar datos que ván a ser encriptados
    int kilobytes;
    sscanf(argv[2], "%d", &kilobytes);
    generate(kilobytes, NOMBRE_ARCHIVO);
    return 0;
}