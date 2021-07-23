if [ "$1" = "--ecb-posix" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_posix src/aes_ecb_posix.cpp
    g++ -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data input.txt 32

    # ./aes_ecb_posix [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_posix input.txt output.bin 1
    ./aes_ecb_posix input.txt output.bin 16

    # Clean
    rm -f aes_ecb_posix
    rm -f generate_data

elif [ "$1" = "--ecb-openmp" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_openmp src/aes_ecb_openmp.cpp
    
    g++ -Ofast -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data input.txt 32

    # ./aes_ecb_openmp [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_openmp input.txt output.bin 1
    ./aes_ecb_openmp input.txt output.bin 16

    # Clean
    rm -f aes_ecb_openmp
    rm -f generate_data

elif [ "$1" = "--cuda" ]; then
    # g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_cuda src/cuda/aes_ecb_cuda.cpp
    nvcc -o aes_ecb_cuda src/cuda/aes_ecb_cuda.cu 
    
    g++ -Ofast -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data src/cuda/input.txt 8


    for ((BLOCKS_GPU = 8 ; BLOCKS_GPU <= 64 ; BLOCKS_GPU *= 2));
    do
        for ((THREADS_PER_BLOCK = 8 ; THREADS_PER_BLOCK <= 64 ; THREADS_PER_BLOCK *= 2));
        do
            # ./aes_cuda [archivo de entrada] [archivo de salida] [numero de bloques] [numero de hilos por bloque]
            ./aes_ecb_cuda input.txt output.bin "$BLOCKS_GPU" "$THREADS_PER_BLOCK"
        done
    done

    # Clean
    rm -f aes_ecb_cuda
    rm -f generate_data

elif [ "$1" = "--clean" ]; then
    rm -f *.txt
    rm -f *.bin
else
    echo "Selecione un flag --ecb-posix o --ecb-openmp"
fi
