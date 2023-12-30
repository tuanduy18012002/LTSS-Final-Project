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
    std::cout << "<------------------------------>" << std::endl;
    //dnn network init
    Network gpu_dnn;
    Layer *conv1 = new gpuConv(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new gpuConv(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    gpu_dnn.add_layer(conv1);
    gpu_dnn.add_layer(pool1);
    gpu_dnn.add_layer(conv2);
    gpu_dnn.add_layer(pool2);
    gpu_dnn.add_layer(fc1);
    gpu_dnn.add_layer(fc2);
    gpu_dnn.add_layer(fc3);
    gpu_dnn.add_layer(relu_conv1);
    gpu_dnn.add_layer(relu_conv2);
    gpu_dnn.add_layer(relu_fc1);
    gpu_dnn.add_layer(relu_fc2);
    gpu_dnn.add_layer(softmax);

    // loss
    Loss *loss = new CrossEntropy;
    gpu_dnn.add_loss(loss);

    gpu_dnn.load_trainnedFile("../data/trained/data-trained.bin");
    timer.Start();
    gpu_dnn.forward(dataset.test_data);
    timer.Stop();
	std::cout << "GPU Forward Time: " << timer.Elapsed() << " ms" << std::endl;
    std::cout << "test accuracy: " << compute_accuracy(gpu_dnn.output(), dataset.test_labels) << std::endl;
    std::cout << "<------------------------------>" << std::endl;
    return EXIT_SUCCESS;
}