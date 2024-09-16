/*****************************************************************************
*  Prelude Handwritten Digits Recognition                                    *
*  Copyright (C) 2024 Brown1729                                              *
*                                                                            *
*  This program is free software; you can redistribute it and/or modify      *
*  it under the terms of the GNU General Public License version 3 as         *
*  published by the Free Software Foundation.                                *
*                                                                            *
*  You should have received a copy of the GNU General Public License         *
*  along with OST. If not, see <http://www.gnu.org/licenses/>.               *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*  @file     trainingProgram.cpp                                             *
*  @brief    Handwritten digits recognition training program                 *
*  @requirements  Eigen: support matrix manipulation                         *
*   MNIST dataset: train-images.idx3-ubyte                                   *
*                  train-labels.idx1-ubyte                                   *
*                  t10k-images.idx3-ubyte                                    *
*                  t10k-labels.idx1-ubyte                                    *
*  @author   Brown1729                                                       *
*  @version  1.3                                                             *
*  @date     2024/9/14                                                       *
*  @license  GNU General Public License (GPL)                                *
*****************************************************************************/

#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <fstream>
#include "Eigen\Dense"
#include <iomanip>

using namespace std;
using namespace Eigen;

/*
 * The defination of hyperparameters.
 */
const double learning_rate = 0.005; // 0.1
const double lambda = 6.0;         // lambda for L2-normalization
const int _TRAINING_SIZE = 60000;  // quantity of data for training
const int _EPOCH = 60;             // epoch
const int _MINI_BATCH_SIZE = 10;   // batch size for stochastic gradient descent
const double _MOMENT = 0.9;        // moment for momentum method
const int _TEST_SIZE = 10000;      // quantity of data for testing
const int _PICTURE_SIZE = 784;     // the size of a digit picture

/*
 * Some preliminary functions, including activating function, the derivate of 
 * activate function with the respect to its variable, and cost function.
 * However, we do not apply cost functions in our final code due to its correctness
 * has been proved true. 
 * The cost functions was previously used as a indicator of the rationality and 
 * accuracy of our model and algorithm.
 */
MatrixXd sigmoid(const MatrixXd & input)
{
    MatrixXd temp(input);
    for (int i = 0; i < temp.rows(); ++i)
    {
        for (int j = 0; j < temp.cols(); ++j)
        {
            temp(i, j) = 1 / (1 + exp(-temp(i, j)));
        }
    }
    return temp;
}

ArrayXXd sigmoid_backward(const ArrayXXd & output)
{
    // hadamard product
    return output * (1 - output);
}

MatrixXd QuadraticCost_delta(const MatrixXd & output, const MatrixXd & label)
{
    return (output - label) * sigmoid_backward(output).matrix();
}

MatrixXd CrossEntropyCost_delta(const MatrixXd & output, const MatrixXd & label)
{
    return (output - label);
}

double CrossEntropyCost(const MatrixXd & output, const MatrixXd & label)
{
    double loss = 0.0;
    for (size_t i = 0; i < output.rows(); ++i)
    {
        double term1 = label(i, 0) * log(output(i, 0));
        double term2 = (1 - label(i, 0)) * log(1 - output(i, 0));
        if (isnan(term1))
            term1 = 0.0;
        if (isnan(term2))
            term2 = 0.0;
        loss -= (term1 + term2);
    }
    return loss;
}

class Layer
    /*
     * The layer means full-connected layer which describes the connecting ways of two layers of neurons.
     * input: input of full-connected layer
     * output: output of full-connected layer
     * W: weights of neurons
     * bias: bias of neurons
     * delta: the delta in backpropagation algorithm, belonging to the neurons of output
     * W_grad: the partial derivative of the loss function with respect to the weight
     * bias_grad: the partial derivative of the loss function with respect to the bias
     * velocity: the velocity in momentum method
     * backward_activator: the pointer to the function used in backpropagation, which
     *                     is the derivative of the forward propagation function with
     *                     respect to the output
     * forward_activator: the pointer to the function used in forward propagation
     */
{
private:
    MatrixXd input, output, W, bias, delta, W_grad, bias_grad, velocity;
    ArrayXXd(*backward_activator)(const ArrayXXd &);
    MatrixXd(*forward_activator)(const MatrixXd &);

public:
    friend class Network;
    Layer(const int input_size, const int output_size,
        MatrixXd(*_forward_activator)(const MatrixXd &),
        ArrayXXd(*_backward_activator)(const ArrayXXd &))
    {
        W = MatrixXd(output_size, input_size);
        velocity = MatrixXd::Zero(output_size, input_size);
        W_grad = MatrixXd::Zero(output_size, input_size);

        /*
        * Initialize the weights and bias with normal distribution.
        */
        static default_random_engine e;
        static normal_distribution<double> u(0, 1 / sqrt(input_size));
        /*
        * Initialize weights with N(0, 1/sqrt(input_size)) can squeezing
        * down the giess distribution makes it more unlikely that
        * our neurons are saturated.
        */
        for (int i = 0; i < W.rows(); ++i)
        {
            for (int j = 0; j < W.cols(); ++j)
            {
                W(i, j) = u(e);
            }
        }
        bias = MatrixXd(output_size, 1);
        bias_grad = MatrixXd::Zero(output_size, 1);
        static normal_distribution<double> b(0, 1);
        /*
        * Initialize bias with N(0,1) because the larger the bias is,
        * the less likely the neurons is to be saturated.
        */
        for (int i = 0; i < bias.rows(); ++i)
        {
            bias(i, 0) = b(e);
        }
        forward_activator = _forward_activator;
        backward_activator = _backward_activator;
    }

    void forward(const MatrixXd & _input)
    {
        /*
        * forward propagation
        */
        input = _input;
        output = forward_activator((W * input) + bias);
    }

    void backward(const MatrixXd & delta_array, const MatrixXd & last_W)
    {
        /*
        * backward propagation
        */
        delta = backward_activator(output).array() * (last_W.transpose() * delta_array).array();
        W_grad += delta * input.transpose();
        bias_grad += delta;
    }

    void update(const int mini_batch_size, const double rate, const double moment)
    {
        /*
        * Update weights and bias after back propagation with momentum method.
        */
        velocity = moment * velocity - rate * W_grad / mini_batch_size;
        W = (1 - rate * lambda / _TRAINING_SIZE) * W + velocity;
        bias = bias - rate * bias_grad / mini_batch_size;
        W_grad.setZero();
        bias_grad.setZero();
    }
};

class Network
{
/*
 * A network combining the layers.
 * layers: a layer container
 * accuracy: current accuracy
 * accuracy5epoch: the accuracy of 5 epochs when training
 * best_accuracy: the best accuracy
 * correct_times: the correct times in testing dataset
 * _delta: the function pointer which calculates layer's delta
 */
private:
    vector<Layer> layers;
    double accuracy = 0, accuracy5epoch = 0, best_accuracy = 0;
    int correct_times = 0;
    MatrixXd(*_delta)(const MatrixXd & output, const MatrixXd & label);

public:
    Network(vector<int> layer_size, MatrixXd(*forward_activator)(const MatrixXd &),
        ArrayXXd(*backward_activator)(const ArrayXXd &),
        MatrixXd(*delta)(const MatrixXd & output, const MatrixXd & label))
    {
        _delta = delta;
        for (int i = 0; i < layer_size.size() - 1; ++i)
        {
            layers.push_back(Layer(
                layer_size[i], layer_size[i + 1],
                forward_activator, backward_activator));
        }
    }

    void predict(const MatrixXd & sample)
    /*
     * Literally the name of the function is 'forward' in contrast to ''backwork' as below.
     * But the operation of 'forward' just like the predicting action. :p
     */
    {
        MatrixXd output = sample;
        for (int i = 0; i < layers.size(); ++i)
        {
            layers[i].forward(output);
            output = layers[i].output;
        }
    }

    void calc_gradient(const MatrixXd & label)
    /*
     * The function is used for calculating the gradient of the loss function with respect 
     * to the weights and bias of each layer. However, the way of processing final layer is
     * a little different from other layers in the backpropagation algorithm.
     */
    {
        int i = layers.size() - 1;
        MatrixXd delta = _delta(layers.back().output, label);
        // MatrixXd delta = (label - layers[i].output);
        layers[i].delta = delta;
        layers[i].W_grad += delta * layers[i].input.transpose();
        layers[i].bias_grad += delta;

        for (i = i - 1; i >= 0; --i)
        {
            layers[i].backward(delta, layers[i + 1].W);
            delta = layers[i].delta;
        }
    }

    void update_weight(const int mini_batch_size, const double rate, const double moment)
    {
        for (int i = 0; i < layers.size(); ++i)
        {
            layers[i].update(mini_batch_size, rate, moment);
        }
    }

    void train_once(const vector<pair<int, int>> & mini_batch, ifstream & samp_file,
        ifstream & label_file, const int mini_batch_size,
        const double rate, const double moment)
    {
        /*
         * A small batch size of training data are put into this function at a time.
         * Then the function will calculate each output together with the deltas of weights
         * and bias. Finally, the function will update weights and bias with the average of 
         * its corresponding deltas.
         */
        MatrixXd sample(_PICTURE_SIZE, 1), label(10, 1);
        unsigned char data = 0;
        for (int i = 0, j = 0; i < mini_batch_size; ++i)
        {
            samp_file.seekg(mini_batch[i].first, ios::beg);
            label_file.seekg(mini_batch[i].second, ios::beg);
            for (j = 0; j < _PICTURE_SIZE; ++j)
            {
                samp_file.read((char *)&data, sizeof(data));
                sample(j, 0) = data / 255.0;
            }
            label_file.read((char *)&data, sizeof(data));
            label(data, 0) = 1;
            predict(sample);
            calc_gradient(label);
            label(data, 0) = 0;
        }
        update_weight(mini_batch_size, rate, moment);
    }

    int get_value(const MatrixXd & mtx) const
    {
        double max = mtx(0, 0);
        int max_index = 0;
        for (int i = 1; i < mtx.rows(); ++i)
        {
            if (mtx(i, 0) > max)
            {
                max = mtx(i, 0);
                max_index = i;
            }
        }
        return max_index;
    }

    void train(vector<pair<int, int>> & training_data, ifstream & samp_file, ifstream & label_file,
        const double rate, const double moment, const int mini_batch_size, const int epoch,
        ifstream & test_samp_file, ifstream & test_label_file, bool isTest)
    {
        mt19937 gen(std::random_device{}());
        /*
         * <random> in C++11
         * Random number generator
         */
        for (int i = 1; i <= epoch; ++i)
        {
            cout << "Start the " << i << "th training session." << endl;
            shuffle(training_data.begin(), training_data.end(), gen);
            /*
             * shuffle data randomly
             */
            for (int j = 0; j < _TRAINING_SIZE; j += mini_batch_size)
            {
                vector<pair<int, int>> mini_batch(
                    training_data.begin() + j,
                    training_data.begin() + min(j + mini_batch_size, _TRAINING_SIZE));
                train_once(mini_batch, samp_file, label_file, mini_batch_size, rate, moment);
            }

            if (isTest)
            {
                test(test_samp_file, test_label_file);
                cout << i << " training session(s) completed, The accuracy is: " << accuracy << endl;
            }

            if (best_accuracy < accuracy)
            {
                best_accuracy = accuracy;
                cout << "Do you want to save the best data?(y/n)" << endl;
                char c;
                cin >> c;
                switch (c)
                {
                case 'y':
                    cout << "Start saving data.." << endl;
                    save_W_bias();
                    cout << "Data have been saved." << endl;
                    break;
                case 'n':
                    break;
                }
                save_W_bias();
            }

            accuracy5epoch += accuracy;
            if (i % 5 == 0)
            {
                if (accuracy5epoch / 5 > accuracy)
                    cout << "The neuron network has been saturated, which ends up the training." << endl;
                accuracy5epoch = 0;
            }
        }
    }

    double test(ifstream & test_samp_file, ifstream & test_label_file)
    {
        accuracy = 0;
        correct_times = 0;

        test_samp_file.seekg(16, ios::beg);
        test_label_file.seekg(8, ios::beg);
        MatrixXd sample(_PICTURE_SIZE, 1), label(10, 1);
        unsigned char data = 0;
        for (int j = 0; j < _TEST_SIZE; ++j)
        {
            for (int k = 0; k < _PICTURE_SIZE; ++k)
            {
                test_samp_file.read((char *)&data, sizeof(unsigned char));
                sample(k, 0) = data / 255.0;
            }
            test_label_file.read((char *)&data, sizeof(data));
            label(data, 0) = 1;
            predict(sample);
            if (get_value(layers.back().output) == get_value(label))
            {
                ++correct_times;
            }
            label(data, 0) = 0;
        }
        accuracy = 1.0 * correct_times / _TEST_SIZE;
        return accuracy;
    }

    void save_W_bias()
    {
        ofstream ofile("deviation_and_bias.dat", ios::binary | ios::out);
        for (int i = 0; i < layers.size(); ++i)
        {
            int j = 0;
            double data = 0;
            for (j = 0; j < layers[i].W.rows(); ++j)
            {
                for (int k = 0; k < layers[i].W.cols(); ++k)
                {
                    data = layers[i].W(j, k);
                    ofile.write((const char *)&data, sizeof(data));
                }
            }
            for (j = 0; j < layers[i].bias.rows(); ++j)
            {
                data = layers[i].bias(j, 0);
                ofile.write((const char *)&data, sizeof(data));
            }
        }
        ofile.close();
    }

    void import_W_bias()
    {
        ifstream ifile("deviation_and_bias.dat", ios::binary);
        for (int i = 0; i < layers.size(); ++i)
        {
            int j = 0;
            double data = 0;
            for (j = 0; j < layers[i].W.rows(); ++j)
            {
                for (int k = 0; k < layers[i].W.cols(); ++k)
                {
                    ifile.read((char *)&data, sizeof(data));
                    layers[i].W(j, k) = data;
                }
            }
            for (j = 0; j < layers[i].bias.rows(); ++j)
            {
                ifile.read((char *)&data, sizeof(data));
                layers[i].bias(j, 0) = data;
            }
        }
        ifile.close();
    }

    int showResult()
    {
        return get_value(layers.back().output);
    }
};

int main()
{
    Network net(vector<int>{_PICTURE_SIZE, 100, 30, 10}, sigmoid, sigmoid_backward, CrossEntropyCost_delta);
    cout << "Network initialzation accomplished." << endl;
    ifstream samp_file("train-images.idx3-ubyte", ios::in | ios::binary);
    ifstream label_file("train-labels.idx1-ubyte", ios::in | ios::binary);
    ifstream test_samp_file("t10k-images.idx3-ubyte", ios::in | ios::binary);
    ifstream test_label_file("t10k-labels.idx1-ubyte", ios::in | ios::binary);
    cout << "Start importing data." << endl;
    net.import_W_bias();
    cout << "Data importing accomplished." << endl;

    /*
     * Below are the code used in training. We comment out those code because 
     * it will overwrite the existent package storing weights and bias when training. 
     */
    
    // vector<pair<int, int>> training_data;
    // for (int i = 0; i < _TRAINING_SIZE; ++i)
    // {
    //     training_data.push_back(pair<int, int>(16 + _PICTURE_SIZE * i, 8 + i));
    // }
    // cout << "Training data reading accomplished." << endl;
    // cout << "Start training..." << endl;
    // net.train(training_data, samp_file, label_file, learning_rate, _MOMENT,
    //           _MINI_BATCH_SIZE, _EPOCH, test_samp_file, test_label_file, true);

    // cout << "Training accomplished. Start testing." << endl;

    /*
     * The code for testing.
     */
    cout << "Start testing..." << endl;
    double accuracy = net.test(test_samp_file, test_label_file);
    cout << "Test accomplished, the accuracy is: " << accuracy << endl;

    samp_file.close();
    label_file.close();
    test_samp_file.close();
    test_label_file.close();
    system("pause");
}