
if [ "$1" = "--ecb-posix" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_posix src/aes_ecb_posix.cpp
    g++ -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data input.txt 64

    # ./aes_ecb_posix [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_posix input.txt output.bin 1
    ./aes_ecb_posix input.txt output.bin 2
    ./aes_ecb_posix input.txt output.bin 4
    ./aes_ecb_posix input.txt output.bin 8
    ./aes_ecb_posix input.txt output.bin 16

    # Clean
    rm -f aes_ecb_posix
    rm -f generate_data

elif [ "$1" = "--ecb-openmp" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_openmp src/aes_ecb_openmp.cpp
    
    g++ -Ofast -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data input.txt 64

    # ./aes_ecb_openmp [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_openmp input.txt output.bin 1
    ./aes_ecb_openmp input.txt output.bin 2
    ./aes_ecb_openmp input.txt output.bin 4
    ./aes_ecb_openmp input.txt output.bin 8
    ./aes_ecb_openmp input.txt output.bin 16

    # Clean
    rm -f aes_ecb_openmp
    rm -f generate_data

elif [ "$1" = "--cuda" ]; then
    # g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_cuda src/cuda/aes_ecb_cuda.cpp
    nvcc -o aes_ecb_cuda src/cuda/aes_ecb_cuda.cu 
    
    g++ -Ofast -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data input.txt 64

    for ((BLOCKS_GPU = 1 ; BLOCKS_GPU <= 64 ; BLOCKS_GPU *= 2));
    do
        for ((THREADS_PER_BLOCK = 1; THREADS_PER_BLOCK <= 64 ; THREADS_PER_BLOCK *= 2));
        do
            # ./aes_cuda [archivo de entrada] [archivo de salida] [numero de bloques] [numero de hilos por bloque]
            ./aes_ecb_cuda input.txt output.bin "$BLOCKS_GPU" "$THREADS_PER_BLOCK"
        done
    done

    # Clean
    rm -f aes_ecb_cuda
    rm -f generate_data
elif [ "$1" = "--mpi" ]; then
    mpiCC -lm -Ofast -std=c++17 -o aes_ecb_mpi src/aes_ecb_mpi.cpp 
    g++ -Ofast -std=c++17 -o generate_data src/generate_data.cpp
    
    # Generar Datos
    ./generate_data input.txt 64
    
    echo "MPI - 1 Nodo"
    time mpirun -np 1 aes_ecb_mpi input.txt output1.bin

    echo "MPI - 2 Nodo"
    time mpirun -np 2 aes_ecb_mpi input.txt output1.bin

    # echo "MPI - 4 Nodo"
    # time mpirun -np 4 aes_ecb_mpi input.txt output1.bin

    # echo "MPI - 8 Nodo"
    # time mpirun -np 8 aes_ecb_mpi input.txt output1.bin

    # Clean
    rm -f aes_ecb_mpi
    rm -f generate_data

elif [ "$1" = "--clean" ]; then
    rm -f *.txt
    rm -f *.bin
else
    echo "Selecione un flag --ecb-posix, --ecb-openmp o --mpi"
fi