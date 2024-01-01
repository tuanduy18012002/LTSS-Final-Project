#define PROJECT_GPU

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/gpu/GpuModel.h"
#include "src/gpu/gpuConv.h"

int main()
{
    MNIST dataset("../data/");
    dataset.read();
    Timer timer;

    
    std::cout << "<------------DEVICE------------>" << std::endl;
    std::cout << "mnist train number: " << dataset.train_data.cols() << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    std::cout << "<---------DEVICE-INFO---------->" << std::endl;
    GPU_Info gpu_info;
    gpu_info.printGpuInfo();
    //dnn network 1 init
    Network gpu_dnn;
    gpu_dnn.add_layer(new gpuConv(1, 28, 28, 6, 5, 5));
    gpu_dnn.add_layer(new MaxPooling(6, 24, 24, 2, 2, 2));
    gpu_dnn.add_layer(new gpuConv(6, 12, 12, 16, 5, 5));
    gpu_dnn.add_layer(new MaxPooling(16, 8, 8, 2, 2, 2));
    gpu_dnn.add_layer(new FullyConnected(MaxPooling(16, 8, 8, 2, 2, 2).output_dim(), 120));
    gpu_dnn.add_layer(new FullyConnected(120, 84));
    gpu_dnn.add_layer(new FullyConnected(84, 10));
    gpu_dnn.add_layer(new ReLU);
    gpu_dnn.add_layer(new ReLU);
    gpu_dnn.add_layer(new ReLU);
    gpu_dnn.add_layer(new ReLU);
    gpu_dnn.add_layer(new Softmax);
        //dnn network 2 init
        Network gpu_dnn2;
        gpu_dnn2.add_layer(new gpuConv_v2(1, 28, 28, 6, 5, 5));
        gpu_dnn2.add_layer(new MaxPooling(6, 24, 24, 2, 2, 2));
        gpu_dnn2.add_layer(new gpuConv_v2(6, 12, 12, 16, 5, 5));
        gpu_dnn2.add_layer(new MaxPooling(16, 8, 8, 2, 2, 2));
        gpu_dnn2.add_layer(new FullyConnected(MaxPooling(16, 8, 8, 2, 2, 2).output_dim(), 120));
        gpu_dnn2.add_layer(new FullyConnected(120, 84));
        gpu_dnn2.add_layer(new FullyConnected(84, 10));
        gpu_dnn2.add_layer(new ReLU);
        gpu_dnn2.add_layer(new ReLU);
        gpu_dnn2.add_layer(new ReLU);
        gpu_dnn2.add_layer(new ReLU);
        gpu_dnn2.add_layer(new Softmax);
            //dnn network 3 init
    Network gpu_dnn3;
    gpu_dnn3.add_layer(new gpuConv_v3(1, 28, 28, 6, 5, 5));
    gpu_dnn3.add_layer(new MaxPooling(6, 24, 24, 2, 2, 2));
    gpu_dnn3.add_layer(new gpuConv_v3(6, 12, 12, 16, 5, 5));
    gpu_dnn3.add_layer(new MaxPooling(16, 8, 8, 2, 2, 2));
    gpu_dnn3.add_layer(new FullyConnected(MaxPooling(16, 8, 8, 2, 2, 2).output_dim(), 120));
    gpu_dnn3.add_layer(new FullyConnected(120, 84));
    gpu_dnn3.add_layer(new FullyConnected(84, 10));
    gpu_dnn3.add_layer(new ReLU);
    gpu_dnn3.add_layer(new ReLU);
    gpu_dnn3.add_layer(new ReLU);
    gpu_dnn3.add_layer(new ReLU);
    gpu_dnn3.add_layer(new Softmax);
    // loss
    Loss *loss = new CrossEntropy;
    gpu_dnn.add_loss(loss);
    gpu_dnn.load_trainnedFile("../data/trained/data-trained.bin");
    std::cout << "<----------VERSION--1---------->" << std::endl;
    timer.Start();
    gpu_dnn.forward(dataset.test_data);
    timer.Stop();
    std::cout << "test accuracy: " << compute_accuracy(gpu_dnn.output(), dataset.test_labels) << std::endl;
    std::cout << "<----------VERSION--2---------->" << std::endl;
    timer.Start();
    gpu_dnn2.forward(dataset.test_data);
    timer.Stop();
    std::cout << "test accuracy: " << compute_accuracy(gpu_dnn.output(), dataset.test_labels) << std::endl;
    std::cout << "<----------VERSION--3---------->" << std::endl;
    timer.Start();
    gpu_dnn3.forward(dataset.test_data);
    timer.Stop();
    std::cout << "test accuracy: " << compute_accuracy(gpu_dnn.output(), dataset.test_labels) << std::endl;
    std::cout << "<------------------------------>" << std::endl;
    return EXIT_SUCCESS;
}