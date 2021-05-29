// from https://github.com/SergeyBel/AES/blob/master/src/AES.cpp

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

using namespace std;

using uint = uint32_t; 
using uint8 = uint8_t;

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

const uint8 sbox[16][16] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
    0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
    0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
    0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
    0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
    0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
    0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
    0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
    0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
    0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
    0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
    0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

const uint8 inv_sbox[16][16] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
    0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
    0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
    0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
    0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
    0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
    0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
    0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
    0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
    0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
    0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
    0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
    0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
    0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
    0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
    0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
};

class AES {
private:
    int Nb;
    int Nk;
    int Nr;

    uint blockBytesLen;

    void SubBytes(vector<vector<uint8>> &state) {
        uint8 t;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                t = state[i][j];
                state[i][j] = sbox[t / 16][t % 16];
            }
        }
    }

    void ShiftRow(vector<vector<uint8>> &state, int i, int n) {    // shift row i on n positions
        vector<uint8> tmp = vector<uint8>(Nb);
        for (int j = 0; j < Nb; j++) {
            tmp[j] = state[i][(j + n) % Nb];
        }
        state[i] = tmp;
        tmp.clear();
    }

    void ShiftRows(vector<vector<uint8>> &state) {
        ShiftRow(state, 1, 1);
        ShiftRow(state, 2, 2);
        ShiftRow(state, 3, 3);
    }

    uint8 xtime(uint8 b) {    // multiply on x
        return uint8((b << 1) ^ (((b >> 7) & 1) * 0x1b));
    }

    uint8 mul_bytes(uint8 a, uint8 b) {
        uint8 p = 0;
        uint8 high_bit_mask = 0x80;
        uint8 high_bit = 0;
        uint8 modulo = 0x1B; /* x^8 + x^4 + x^3 + x + 1 */
        for (int i = 0; i < 8; i++) {
            if (b & 1) {
                p ^= a;
            }
            high_bit = a & high_bit_mask;
            a <<= 1;
            if (high_bit) {
                a ^= modulo;
            }
            b >>= 1;
        }
        return p;
    }

    void MixColumns(vector<vector<uint8>> &state) {
        vector<uint8> temp = vector<uint8>(4);
        for(int i = 0; i < 4; ++i) {
            for(int j = 0; j < 4; ++j) {
                temp[j] = state[j][i]; //place the current state column in temp
            }
            MixSingleColumn(temp); //mix it using the wiki implementation
            for(int j = 0; j < 4; ++j) {
                state[j][i] = temp[j]; //when the column is mixed, place it back into the state
            }
        }
        temp.clear();
    }

    void MixSingleColumn(vector<uint8> &r) {
        vector<uint8> a(4, 0);
        vector<uint8> b(4, 0);
        /* The array 'a' is simply a copy of the input array 'r'
        * The array 'b' is each element of the array 'a' multiplied by 2
        * in Rijndael's Galois field
        * a[n] ^ b[n] is element n multiplied by 3 in Rijndael's Galois field */ 
        for(int c = 0; c < 4; c++) {
            a[c] = r[c];
            /* h is 0xff if the high bit of r[c] is set, 0 otherwise */
            uint8 h = (uint8)((signed char)r[c] >> 7); /* arithmetic right shift, thus shifting in either zeros or ones */
            b[c] = r[c] << 1; /* implicitly removes high bit because b[c] is an 8-bit char, so we xor by 0x1b and not 0x11b in the next line */
            b[c] ^= 0x1B & h; /* Rijndael's Galois field */
        }
        r[0] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1]; /* 2 * a0 + a3 + a2 + 3 * a1 */
        r[1] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2]; /* 2 * a1 + a0 + a3 + 3 * a2 */
        r[2] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3]; /* 2 * a2 + a1 + a0 + 3 * a3 */
        r[3] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0]; /* 2 * a3 + a2 + a1 + 3 * a0 */
    }

    void AddRoundKey(vector<vector<uint8>> &state, vector<uint8> &key, int idx) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                state[i][j] = state[i][j] ^ key[idx + i + 4 * j];
            }
        }
    }

    void SubWord(vector<uint8> &a) {
        for (int i = 0; i < 4; i++) {
            a[i] = sbox[a[i] / 16][a[i] % 16];
        }
    }

    void RotWord(vector<uint8> &a) {
        uint8 c = a[0];
        a[0] = a[1];
        a[1] = a[2];
        a[2] = a[3];
        a[3] = c;
    }

    void XorWords(vector<uint8> &a, vector<uint8> &b, vector<uint8> &c) {
        for (int i = 0; i < 4; i++) {
            c[i] = a[i] ^ b[i];
        }
    }

    void Rcon(vector<uint8> &a, int n) {
        uint8 c = 1;
        for (int i = 0; i < n - 1; i++) {
            c = xtime(c);
        }
        a[0] = c;
        a[1] = a[2] = a[3] = 0;
    }

    void InvSubBytes(vector<vector<uint8>> &state) {
        uint8 t;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                t = state[i][j];
                state[i][j] = inv_sbox[t / 16][t % 16];
            }
        }
    }

    void InvMixColumns(vector<vector<uint8>> &state) {
        uint8 s[4], s1[4];
        for (int j = 0; j < Nb; j++) {
            for (int i = 0; i < 4; i++) {
                s[i] = state[i][j];
            }
            s1[0] = mul_bytes(0x0e, s[0]) ^ mul_bytes(0x0b, s[1]) ^ mul_bytes(0x0d, s[2]) ^ mul_bytes(0x09, s[3]);
            s1[1] = mul_bytes(0x09, s[0]) ^ mul_bytes(0x0e, s[1]) ^ mul_bytes(0x0b, s[2]) ^ mul_bytes(0x0d, s[3]);
            s1[2] = mul_bytes(0x0d, s[0]) ^ mul_bytes(0x09, s[1]) ^ mul_bytes(0x0e, s[2]) ^ mul_bytes(0x0b, s[3]);
            s1[3] = mul_bytes(0x0b, s[0]) ^ mul_bytes(0x0d, s[1]) ^ mul_bytes(0x09, s[2]) ^ mul_bytes(0x0e, s[3]);
            for (int i = 0; i < 4; i++) {
                state[i][j] = s1[i];
            }
        }
    }

    void InvShiftRows(vector<vector<uint8>> &state) {
        ShiftRow(state, 1, Nb - 1);
        ShiftRow(state, 2, Nb - 2);
        ShiftRow(state, 3, Nb - 3);
    }

    vector<uint8> PaddingNulls(vector<uint8> &in, uint inLen, uint alignLen) {
        uint8 *alignIn = new uint8[alignLen];
        for(uint i = 0; i < inLen; ++i) {
            alignIn[i] = in[i];
        }
        memset(alignIn + inLen, 0x00, alignLen - inLen);
        vector<uint8> to_return = vector<uint8>(alignLen);
        for(uint i = 0; i < alignLen; ++i) {
            to_return[i] = alignIn[i];
        }
        return to_return;
    }

    uint GetPaddingLength(uint len) {
        uint lengthWithPadding =  (len / blockBytesLen);
        if (len % blockBytesLen) {
            lengthWithPadding++;
        }
        lengthWithPadding *=  blockBytesLen;
        return lengthWithPadding;
    }

    void KeyExpansion(vector<uint8> &key, vector<uint8> &w) {
        vector<uint8> temp = vector<uint8>(4);
        vector<uint8> rcon = vector<uint8>(4);
        int i = 0;
        while (i < 4 * Nk) {
            w[i] = key[i];
            i++;
        }
        i = 4 * Nk;
        while (i < 4 * Nb * (Nr + 1)) {
            temp[0] = w[i - 4 + 0];
            temp[1] = w[i - 4 + 1];
            temp[2] = w[i - 4 + 2];
            temp[3] = w[i - 4 + 3];
            if (i / 4 % Nk == 0) {
                RotWord(temp);
                SubWord(temp);
                Rcon(rcon, i / (Nk * 4));
                XorWords(temp, rcon, temp);
            } else if (Nk > 6 && i / 4 % Nk == 4) {
                SubWord(temp);
            }
            w[i + 0] = w[i - 4 * Nk] ^ temp[0];
            w[i + 1] = w[i + 1 - 4 * Nk] ^ temp[1];
            w[i + 2] = w[i + 2 - 4 * Nk] ^ temp[2];
            w[i + 3] = w[i + 3 - 4 * Nk] ^ temp[3];
            i += 4;
        }
        rcon.clear();
        temp.clear();
    }

    void EncryptBlock(vector<uint8> &in, vector<uint8> &out, vector<uint8> &roundKeys, int idx) {
        vector<vector<uint8>> state = vector<vector<uint8>>(4, vector<uint8>(4));
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                state[i][j] = in[idx + i + 4 * j];
            }
        }

        AddRoundKey(state, roundKeys, 0);
        for (int round = 1; round <= Nr - 1; round++) {
            SubBytes(state);
            ShiftRows(state);
            MixColumns(state);
            AddRoundKey(state, roundKeys, round * 4 * Nb);
        }
        SubBytes(state);
        ShiftRows(state);
        AddRoundKey(state, roundKeys, Nr * 4 * Nb);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                out[idx + i + 4 * j] = state[i][j];
            }
        }
        state.clear();
    }

    void DecryptBlock(vector<uint8> &in, vector<uint8> &out, vector<uint8> &roundKeys, int idx) {
        vector<vector<uint8>> state = vector<vector<uint8>>(4, vector<uint8>(4));
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                state[i][j] = in[idx+i + 4 * j];
            }
        }
        AddRoundKey(state, roundKeys,  Nr * 4 * Nb);
        for (int round = Nr - 1; round >= 1; round--) {
            InvSubBytes(state);
            InvShiftRows(state);
            AddRoundKey(state, roundKeys, round * 4 * Nb);
            InvMixColumns(state);
        }
        InvSubBytes(state);
        InvShiftRows(state);
        AddRoundKey(state, roundKeys, 0);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < Nb; j++) {
                out[idx+i + 4 * j] = state[i][j];
            }
        }
        state.clear();
    }

public:
    AES(int keyLen = 256) {
        this->Nb = 4;
        switch (keyLen) {
            case 128:
                this->Nk = 4;
                this->Nr = 10;
                break;
            case 192:
                this->Nk = 6;
                this->Nr = 12;
                break;
            case 256:
                this->Nk = 8;
                this->Nr = 14;
                break;
            default:
                throw "Incorrect key length";
        }
        blockBytesLen = 4 * this->Nb * sizeof(uint8);
    }

    vector<uint8> EncryptECB(vector<uint8> &in, uint inLen, vector<uint8> &key, uint &outLen) {
        outLen = GetPaddingLength(inLen);
        vector<uint8> alignIn  = PaddingNulls(in, inLen, outLen);
        vector<uint8> out = vector<uint8>(outLen);
        vector<uint8> roundKeys = vector<uint8>(4 * Nb * (Nr + 1));
        KeyExpansion(key, roundKeys);
        for (uint i = 0; i < outLen; i+= blockBytesLen) {
            EncryptBlock(alignIn, out, roundKeys, i);
        }
        alignIn.clear();
        roundKeys.clear();
        return out;
    }

    vector<uint8> DecryptECB(vector<uint8> &in, uint inLen, vector<uint8> &key) {
        vector<uint8> out = vector<uint8>(inLen);
        vector<uint8> roundKeys = vector<uint8>(4 * Nb * (Nr + 1));
        KeyExpansion(key, roundKeys);
        for (uint i = 0; i < inLen; i += blockBytesLen) {
            DecryptBlock(in, out, roundKeys, i);
        }
        roundKeys.clear();
        return out;
    }
};

void print_hex_array(const vector<uint8> &a) {
    for (int i = 0; i < (int) a.size(); i++) {
        printf("%02x ", a[i]);
    }
    printf("\n");
}

vector<vector<uint8>> text_to_hex(const string &text) {
    int blocks_size = 16;
    int n = (int) text.size();
    int blocks = (n+blocks_size-1)/blocks_size;
    vector<vector<uint8>> texthex = vector<vector<uint8>>(blocks, vector<uint8>(blocks_size, 0));
    int idx_block = 0;
    int block = 0;
    for(int i = 0; i < n; ++i) {
        texthex[block][idx_block] = uint8(text[i]);
        ++idx_block;
        if(idx_block == 16) {
            idx_block = 0;
            block++;
        }
    }
    return texthex;
}

vector<uint8> key_to_hex(const string &key) {
    int n = (int) key.size();
    assert(0 <= n && n <= 32);
    vector<uint8> keyhex = vector<uint8>(32, 0);
    for(int i = 0; i < n; ++i) {
        keyhex[i] = uint8(key[i]);
    }
    return keyhex;
}

void build_hex(string &text, string &key) {
    text_hex = text_to_hex(text);
    key_hex = key_to_hex(key);
}

// Instancia global del AES
AES aes(256);

const uint BLOCK_BYTES_LENGTH = 16 * sizeof(uint8);

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

    string text = "";
    string key = "admin1234"; // maximo 32 caracteres

    ifstream input;
    input.open(NOMBRE_ARCHIVO);

    text.reserve(1e8);

    string line;
    while(getline(input, line)) {
        text += line;
    }

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
    ofstream file;
    file.open(ARCHIVO_SALIDA);
    if(!file) {
        cout << "No se puedo abrir el archivo: \"" << ARCHIVO_SALIDA << "\"" << endl;
        exit(1);
    }

    // Escribir el texto encriptado en el archivo ARCHIVO_SALIDA
    for(vector<uint8> &hexa: cipher_text) {
        for(uint8 &val: hexa) {
           if(val < 16) file << '0';
            file << hex << (int) val;
        }
        file << '\n';
    }
    file.close();
    
    return 0;
}
