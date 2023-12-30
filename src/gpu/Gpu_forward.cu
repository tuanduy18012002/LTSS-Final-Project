#include "Gpu_forward.h"

void GpuForward::Conv_forward(const Matrix& bottom, int Blocksize){
  dim3 blocksize;
  blocksize.x = Blocksize;
  printf("Gpu_Convolution_Kernel %d", blocksize.x);
}