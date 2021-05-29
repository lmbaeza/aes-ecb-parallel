if [ "$1" = "--ecb-posix" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_posix aes_ecb_posix.cpp
    
    # ./aes_ecb_posix [archivo de entrada] [archivo de salida] [numero de hilos]
    ./aes_ecb_posix input.txt output.bin 1
    ./aes_ecb_posix input.txt output.bin 16
elif [ "$1" = "--ecb-openmp" ]; then
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes_ecb_openmp aes_ecb_openmp.cpp && ./aes_ecb_openmp
else
    g++ -D_DEFAULT_SOURCE -fopenmp -lm -Ofast -std=c++17 -o aes aes.cpp && ./aes
fi
