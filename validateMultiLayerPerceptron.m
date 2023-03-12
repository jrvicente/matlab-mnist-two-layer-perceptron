function [correctlyClassified, classificationErrors] = validateMultiLayerPerceptron(option, inputValuesTest, labels)
% validateMultiLayerPerceptron(activationFunction, hiddenWeights, outputWeights, inputValues, labels)
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

    testSetSize = size(inputValuesTest, 2);
    classificationErrors = 0;
    correctlyClassified = 0;
    
    for n = 1: testSetSize
        inputVector = inputValuesTest(:, n);
        outputVector = evaluateMultiLayerPerceptron(option, inputVector);
        
        class = decisionRule(outputVector);
        if class == labels(n) + 1
            correctlyClassified = correctlyClassified + 1;
        else
            classificationErrors = classificationErrors + 1;
        end;
    end;
end

function class = decisionRule(outputVector)
% decisionRule Model based decision rule.
%
% INPUT:
% outputVector      : Output vector of the network.
%
% OUTPUT:
% class             : Class the vector is assigned to.
%

    max = 0;
    class = 1;
    for i = 1: size(outputVector, 1)
        if outputVector(i) > max
            max = outputVector(i);
            class = i;
        end;
    end;
end

function outputVector = evaluateMultiLayerPerceptron(option, inputVector)
% evaluateTwoLayerPerceptron Evaluate two-layer perceptron given by the
% weights using the given activation function.
%
% INPUT:
% activationFunction             : Activation function used in both layers.
% hiddenWeights                  : Weights of hidden layer.
% outputWeights                  : Weights for output layer.
% inputVector                    : Input vector to evaluate.
%
% OUTPUT:
% outputVector                   : Output of the perceptron.
% 
%    outputVector = activationFunction(outputWeights*activationFunction(hiddenWeights*inputVector));
    
    alpha = option.activationFunctionAlpha;
    activationFunction = option.activationFunction;
    layerSizes = option.layerSizes;
    numLayers = length(layerSizes);
    layerWeights = option.layerWeights;
    layerBias = option.layerBias;
    
    layerVector = cell(1,numLayers - 1);
    layerVector{1} = inputVector;
    for L = 1 : numLayers - 1
%         layerVector{L+1} = activationFunction(layerWeights{L}*layerVector{L} + layerBias{L});
        layerVector{L+1} = activationFunction(layerWeights{L}*layerVector{L} + layerBias{L}, alpha);
    end
    
    outputVector = layerVector{end};
end