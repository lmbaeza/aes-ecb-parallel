# Aes Modo ECB & CTR

Paralelizar el Modo ECB & CTR del aes

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
