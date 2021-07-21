%%writefile file_handler_cuda.h
#ifndef FILE_HANDLER
#define FILE_HANDLER

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<cstdint>
using namespace std;

using uint = uint32_t;
using uint8 = uint8_t;

void write_file(const string &filename, uint8 (*data)[16], int x) {
    ofstream file;
    file.open(filename);
    if(!file) {
        cout << "No se puedo abrir el archivo: \"" << filename << "\"" << endl;
        exit(1);
    }

    // Escribir el texto encriptado en el archivo filename
    for(int i = 0; i < x; ++i) {
        for(int j = 0; j < 16; ++j) {
            uint8 val = data[i][j];
            if(val < 16) file << '0';
            file << hex << (int) val;
        }
        file << '\n';
    }
    file.close();
}

void read_file_to_string(const string &filename, string &text) {
    ifstream input;
    input.open(filename);
    text.reserve(1e8);

    string line;
    while(getline(input, line)) {
        text += line;
    }
}

#endif