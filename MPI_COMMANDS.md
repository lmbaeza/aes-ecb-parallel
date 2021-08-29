# Comandos MPI


```bash
# (Master)
sudo ssh user@cluster-node
exit

# (Node)
sudo ssh user@cluster-master
exit

# (Master Solo Una Vez)
cd ~/.ssh
ssh-keygen -t rsa -b 4096 -C "user@email.com"
cp id_rsa.pub authorized_keys

# (Master)
scp id_rsa id_rsa.pub user@cluster-node

# (Node)
mv id_rsa id_rsa.pub ~/.ssh
cd ~/.ssh
cp id_rsa.pub authorized_keys

# Testear Conectividad SSH
# (Master)
sudo ssh user@cluster-node
exit
# (Node)
sudo ssh user@cluster-master
exit

# (Master)
sudo apt-get update
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev -y
sudo apt-get install nfs-kernel-server -y
sudo apt-get install nfs-common -y
sudo chmod 777 /etc/exports
echo "/home/user *(rw,sync,no_subtree_check)" > /etc/exports
sudo service nfs-kernel-server restart
sudo exportfs -a

# (Node)
sudo apt-get update
sudo apt-get install nfs-common -y
sudo mount -t nfs ip_del_master:/home/user /home/user
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev -y

# (Master)
vim /home/user/mpi_hosts

# Copiar
# -------------------------------------
# The Hostfile for Open MPI
# The master node, 'slots=2' is used because it is a dual-processor machine.
localhost slots=1
# The following slave nodes are single processor machines:
azurenode1@cluster-node1
# -------------------------------------

# (Master and Node)
sudo apt install g++ -y
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev -y
```


# Run MPI


```bash
git clone https://github.com/lmbaeza/aes-ecb-parallel.git

mpiCC -lm -Ofast -std=c++17 -o aes_ecb_mpi aes-ecb-parallel/src/aes_ecb_mpi.cpp
g++ -Ofast -std=c++17 -o generate_data aes-ecb-parallel/src/generate_data.cpp

# Generate 1MB
rm -f input.txt output1.bin
./generate_data input.txt "1"

time mpirun -np 1 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 2 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 4 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 8 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin

# Generate 16MB
rm -f input.txt output1.bin
./generate_data input.txt "16"

time mpirun -np 1 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 2 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 4 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 8 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin

# Generate 16MB
rm -f input.txt output1.bin
./generate_data input.txt "64"

time mpirun -np 1 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 2 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 4 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
time mpirun -np 8 --hostfile mpi-hosts ./aes_ecb_mpi input.txt output1.bin
```