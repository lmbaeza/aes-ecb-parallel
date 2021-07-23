# Aes Modo ECB & CTR

Paralelizar el Modo ECB del aes

----

**Universidad Nacional de Colombia - Sede Bogotá**

 _**Computación Paralela**_

 **Docente:**   César Pedraza Bonilla

 **Estudiantes:**
 * Luis Miguel Báez Aponte - lmbaeza@unal.edu.co
 * Bryan David Velandia Parada - bdvelandiap@unal.edu.co
 * Camilo Ernesto Vargas Romero - camevargasrom@unal.edu.co


# Descargar

* Descargar Rama `develop`
```shell
$ git clone --single-branch --branch=develop https://github.com/lmbaeza/aes-ecb-parallel.git
```

* Darle permiso de Ejecución

```shell
$ cd aes-ecb-parallel
$ sudo chmod 777 run.sh 
```

# Ejecutar

### Ejecutar `aes_ecb_posix.cpp`

```shell
$ ./run.sh --ecb-posix
```

### Ejecutar `aes_ecb_openmp.cpp`

```shell
$ ./run.sh --ecb-openmp
```

### Ejecutar `Cuda`

```shell
$ ./run.sh --cuda
```
O puede correrlo Manualmente

```shell
nvcc -o aes_ecb_cuda src/cuda/aes_ecb_cuda.cu 
g++ -Ofast -std=c++17 -o generate_data src/generate_data.cpp
./generate_data input.txt 16

for ((BLOCKS_GPU = 1 ; BLOCKS_GPU <= 64 ; BLOCKS_GPU *= 2));
    do
        for ((THREADS_PER_BLOCK = 1; THREADS_PER_BLOCK <= 64 ; THREADS_PER_BLOCK *= 2));
        do
            ./aes_ecb_cuda input.txt output.bin "$BLOCKS_GPU" "$THREADS_PER_BLOCK"
        done
    done
```
