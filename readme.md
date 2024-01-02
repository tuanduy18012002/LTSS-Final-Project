Premake : 
```shell
mkdir build
cd build
cmake ..
make
```
The data is already trained in the data_trained.bin file, we just need :
- Run `./cpu_main` to run CPU model
- Run `./gpu_main` to run GPU model (Cuda version)