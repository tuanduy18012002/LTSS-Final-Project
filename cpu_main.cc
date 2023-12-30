#define PROJECT_CPU

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
#include "src/cpu/Timer.h"

int main()
{
    MNIST dataset("../data/");
    dataset.read();
    Timer timer;

    
    std::cout << "<-------------HOST------------->" << std::endl;
    std::cout << "mnist train number: " << dataset.train_data.cols() << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    std::cout << "<------------------------------>" << std::endl;
    //dnn network init
    Network cpu_dnn;
    cpu_dnn.add_layer(new Conv(1, 28, 28, 6, 5, 5));
    cpu_dnn.add_layer(new MaxPooling(6, 24, 24, 2, 2, 2));
    cpu_dnn.add_layer(new Conv(6, 12, 12, 16, 5, 5));
    cpu_dnn.add_layer(new MaxPooling(16, 8, 8, 2, 2, 2));
    cpu_dnn.add_layer(new FullyConnected(MaxPooling(16, 8, 8, 2, 2, 2).output_dim(), 120));
    cpu_dnn.add_layer(new FullyConnected(120, 84));
    cpu_dnn.add_layer(new FullyConnected(84, 10));
    cpu_dnn.add_layer(new ReLU);
    cpu_dnn.add_layer(new ReLU);
    cpu_dnn.add_layer(new ReLU);
    cpu_dnn.add_layer(new ReLU);
    cpu_dnn.add_layer(new Softmax);
    // loss
    Loss *loss = new CrossEntropy;
    cpu_dnn.add_loss(loss);

    cpu_dnn.load_trainnedFile("../data/trained/data-trained.bin");
    timer.Start();
    cpu_dnn.forward(dataset.test_data);
    timer.Stop();
	std::cout << "CPU Forward Time: " << timer.Elapsed() << " ms" << std::endl;
    std::cout << "test accuracy: " << compute_accuracy(cpu_dnn.output(), dataset.test_labels) << std::endl;
    std::cout << "<------------------------------>" << std::endl;
    return EXIT_SUCCESS;
}