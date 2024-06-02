# Homework 2
## CUDA Matrix Transposition

## Install
First clone the repository
```
git clone https://github.com/SecondarySkyler/gpu-hw2.git
```
Enter the directory
```
cd gpu-hw2
```
Compile the executable
```
make all
```

## Usage
If your PC has an NVIDIA GPU you can run the executable locally as follows:
```
./transpose <NUMBER>
```
Instead, if you're using the Marzola cluster, the ```run.sh``` file will do the job. <br>
The only thing needed is to specify the dimension of the matrix (default is 2<sup>10</sup>).
You can run it by simply using:
```
sbatch run.sh
```

