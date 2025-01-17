%{
# Recognizing Handwritten Digits using a Two-layer Perceptron

This repository contains code corresponding to the seminar paper:

D. Stutz. **Introduction to Neural Networks.** Seminar Report, Human Language Technology and Pattern Recognition Group, RWTH Aachen University, 2014.

Advisor: Pavel Golik

**Update:** The code can be adapted to allow mini-batch training as done in [this fork](https://github.com/Myasuka/matlab-mnist-two-layer-perceptron).

## MNIST Dataset

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) provides a training set of 60,000 handwritten digits and a validation set of 10,000 handwritten digits. The images have size 28 x 28 pixels. Therefore, when using a two-layer perceptron, we need 28 x 28 = 784 input units and 10 output units (representing the 10 different digits).

The methods `loadMNISTImages` and `loadMNISTLaels` are used to load the MNIST dataset as it is stored in a special file format. The methods can be found online at [http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset](http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset).

## Methods and Usage

The main method to train the two-layer perceptron is `trainStochasticSquaredErrorTwoLayerPerceptron`. The method applies stochastic training (or to be precise a stochastic variant of mini-batch training) using the sum-of-squared error function and the error backpropagation algorithm.

	function [hiddenWeights, outputWeights, error] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, numberOfHiddenUnits, inputValues, targetValues, epochs, batchSize, learningRate)
	% trainStochasticSquaredErrorTwoLayerPerceptron Creates a two-layer perceptron
	% and trains it on the MNIST dataset.
	%
	% INPUT:
	% activationFunction             : Activation function used in both layers.
	% dActivationFunction            : Derivative of the activation
	% function used in both layers.
	% numberOfHiddenUnits            : Number of hidden units.
	% inputValues                    : Input values for training (784 x 60000)
	% targetValues                   : Target values for training (1 x 60000)
	% epochs                         : Number of epochs to train.
	% batchSize                      : Plot error after batchSize images.
	% learningRate                   : Learning rate to apply.
	%
	% OUTPUT:
	% hiddenWeights                  : Weights of the hidden layer.
	% outputWeights                  : Weights of the output layer.

The above method requires the activation function used for both the hidden and the output layer to be given as parameter. I used the logistic sigmoid activation function:

	function y = logisticSigmoid(x)
	% simpleLogisticSigmoid Logistic sigmoid activation function
	% 
	% INPUT:
	% x     : Input vector.
	%
	% OUTPUT:
	% y     : Output vector where the logistic sigmoid was applied element by
	% element.
	%
	
In addition, the error backpropagation algorithm needs the derivative of the used activation function:

	function y = dLogisticSigmoid(x)
	% dLogisticSigmoid Derivative of the logistic sigmoid.
	% 
	% INPUT:
	% x     : Input vector.
	%
	% OUTPUT:
	% y     : Output vector where the derivative of the logistic sigmoid was
	% applied element by element.
	%
	
The method `applyStochasticSquaredErrorTwoLayerPerceptronMNIST` uses both the training method seen above and the method `validateTwoLayerPerceptron` to evaluate the performance of the two-layer perceptron:

	function [correctlyClassified, classificationErrors] = validateTwoLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels)
	% validateTwoLayerPerceptron Validate the twolayer perceptron using the
	% validation set.
	%
	% INPUT:
	% activationFunction             : Activation function used in both layers.
	% hiddenWeights                  : Weights of the hidden layer.
	% outputWeights                  : Weights of the output layer.
	% inputValues                    : Input values for training (784 x 10000).
	% labels                         : Labels for validation (1 x 10000).
	%
	% OUTPUT:
	% correctlyClassified            : Number of correctly classified values.
	% classificationErrors           : Number of classification errors.
	% 
	
## License

License for source code corresponding to:

D. Stutz. **Introduction to Neural Networks.** Seminar Report, Human Language Technology and Pattern Recognition Group, RWTH Aachen University, 2014.

Copyright (c) 2014-2018 David Stutz

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
%}


% function [] = applyStochasticSquaredErrorMultiLayerPerceptronMNIST()
%applyStochasticSquaredErrorTwoLayerPerceptronMNIST Train the two-layer
%perceptron using the MNIST dataset and evaluate its performance.

    % Load MNIST.
    inputValues = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    
    % Transform the labels to correct target values.
    targetValues = zeros(10, size(labels, 1));
    for n = 1: size(labels, 1)
        targetValues(labels(n) + 1, n) = 1;
    end;
    
    % Choose activation function.
    activationFunction = @leakyReLU;
    dActivationFunction = @dLeakyReLU;
    
    % Choose appropriate parameters. 
    learningRate = 0.1; % original value 0.1
    
    % Choose form of MLP:
    layerSizes = [size(inputValues, 1), 50, 20, 10]; % First layer is the input and last layer is the output layer.
        % Original values [748, 700, 10] two layers;
    
    % Choose batch size and epochs. Remember there are 60k input values.
    batchSize = 100;  % origin value 100
    epochs = 1000; % original iteration number 500
    
    fprintf('Input vector length: %d \n', layerSizes(1));
    fprintf('Output vector length: %d \n', layerSizes(end));
    fprintf('Number of neural layers: %d \n', length(layerSizes)-2); 
    fprintf(['Train perceptron with (', num2str(layerSizes(2:end-1)), ') layer sizes. \n']);
    fprintf('Training sample size: %d \n', size(inputValues, 2));
    fprintf('Training batch size: %d \n', batchSize);
    fprintf(['Learning rate: ', num2str(learningRate), '\n']);
    
    option.activationFunctionAlpha = 0.1;
    option.activationFunction = activationFunction;
    option.dActivationFunction = dActivationFunction;
    option.layerSizes = layerSizes;
    option.epochs = epochs;
    option.batchSize = batchSize;
    option.learningRate = learningRate;
    
    tic
    fprintf('Training epochs %d progressing: \n', epochs);
    [optionTrainedMultiLayer, error] = trainStochasticSquaredErrorMultiLayerPerceptron(option, inputValues, targetValues);
%    [layerWeights, lastlayerWeights, error] = trainStochasticSquaredErrorTwoLayerPerceptron(activationFunction, dActivationFunction, layerSizes, inputValues, targetValues, epochs, batchSize, learningRate);
    toc
    
    % Load validation set.
    inputValuesTest = loadMNISTImages('t10k-images.idx3-ubyte');
    labels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    
    % Choose decision rule.
    fprintf('Validation:\n');
    fprintf('Validation sample size: %d \n', size(inputValuesTest, 2));
       
    [correctlyClassified, classificationErrors] = validateMultiLayerPerceptron(optionTrainedMultiLayer, inputValuesTest, labels);
    
    fprintf('Classification errors: %d\n', classificationErrors);
    fprintf('Correctly classified: %d\n', correctlyClassified);
    fprintf(['Classified correctly: ', num2str(round(correctlyClassified/(correctlyClassified + classificationErrors)*100, 2)), ' %% \n']);
% end