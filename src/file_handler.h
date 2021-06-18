using uint = uint32_t; 
using uint8 = uint8_t;

void write_file(const string &filename, vector<vector<uint8>> &data) {
    ofstream file;
    file.open(filename);
    if(!file) {
        cout << "No se puedo abrir el archivo: \"" << filename << "\"" << endl;
        exit(1);
    }

    // Escribir el texto encriptado en el archivo filename
    for(vector<uint8> &hexa: data) {
        for(uint8 &val: hexa) {
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