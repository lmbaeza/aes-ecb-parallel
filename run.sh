if [ "$1" = "--ecb-posix" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_posix src/aes_ecb_posix.cpp
    
    # ./aes_ecb_posix [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_posix input.txt output.bin 1
    ./aes_ecb_posix input.txt output.bin 16

    rm -f aes_ecb_posix
elif [ "$1" = "--ecb-openmp" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_openmp src/aes_ecb_openmp.cpp
    
    # ./aes_ecb_openmp [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_openmp input.txt output.bin 1
    ./aes_ecb_openmp input.txt output.bin 16

    rm -f aes_ecb_openmp
else
    echo "Selecione un flag --ecb-posix o --ecb-openmp"
fi
