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
    Layer* conv1 = new Conv(1, 28, 28, 6, 5, 5); 
    Layer* relu1 = new ReLU;
    Layer* pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer* conv2 = new Conv(6, 12, 12, 16, 5, 5);
    Layer* relu2 = new ReLU;
    Layer* pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer* conv3 = new Conv(16, 4, 4, 120, 4, 4);
    Layer* relu3 = new ReLU;
    Layer* fc1 = new FullyConnected(120, 84);
    Layer* relu4 = new ReLU;
    Layer* fc2 = new FullyConnected(84, 10);
    Layer* softmax = new Softmax;

    gpu_dnn.add_layer(conv1);
	gpu_dnn.add_layer(relu1);
	gpu_dnn.add_layer(pool1);
	gpu_dnn.add_layer(conv2);
	gpu_dnn.add_layer(relu2);
	gpu_dnn.add_layer(pool2);
	gpu_dnn.add_layer(conv3);
	gpu_dnn.add_layer(relu3);
	gpu_dnn.add_layer(fc1);
	gpu_dnn.add_layer(relu4);
	gpu_dnn.add_layer(fc2);
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