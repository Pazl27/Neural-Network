package org.example;

import java.util.Random;

public class NeuralNetworkXOR {

    public static int numInputs = 2;
    public static int numHiddenNodes = 2;
    public static int numOutputs = 1;
    public static int numTrainingSets = 4;

    private static double initWeights(){
        return new Random().nextDouble();
    }

    private static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    private static double sigmoidDerivative(double x){
        return x * (1 - x);
    }

    private static void shuffle(int[] array){
        Random rand = new Random();
        int n = array.length;

        if (n > 1) {
            for (int i = 0; i < n - 1; i++) {
                int j = i + rand.nextInt(n - i);
                int temp = array[j];
                array[j] = array[i];
                array[i] = temp;
            }
        }
    }

    private static void backpropagation(double[][] trainingOutputs, int i, double[] outputLayer, double[][] outputWeights, double[] hiddenLayer, double[] outputLayerBias, double learningRate, double[] hiddenLayerBias, double[][] hiddenWeights, double[][] trainingInputs) {
        //Change in output weights
        double[] deltaOutput= new double[numOutputs];

        for(int j = 0; j < numOutputs; j++){
            double error = (trainingOutputs[i][j] - outputLayer[j]);
            deltaOutput[j] = error * sigmoidDerivative(outputLayer[j]);
        }

        //Compute change in hidden weights
        double[] deltaHidden = new double[numHiddenNodes];
        for(int j = 0; j < numHiddenNodes; j++){
            double error = 0.0f;
            for(int k = 0; k < numOutputs; k++){
                error += deltaOutput[k] * outputWeights[j][k];
            }
            deltaHidden[j] = error * sigmoidDerivative(hiddenLayer[j]);
        }


        //Update output weights
        for(int j = 0; j < numOutputs; j++){
            outputLayerBias[j] += deltaOutput[j] * learningRate;
            for(int k = 0; k < numHiddenNodes; k++){
                outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * learningRate;
            }
        }

        //Update hidden weights
        for(int j = 0; j < numHiddenNodes; j++){
            hiddenLayerBias[j] += deltaHidden[j] * learningRate;
            for(int k = 0; k < numInputs; k++){
                hiddenWeights[k][j] += trainingInputs[i][k] * deltaHidden[j] * learningRate;
            }
        }
    }

    private static void forwardPass(double[] hiddenLayerBias, double[][] hiddenWeights, double[] hiddenLayer, double[] outputLayerBias, double[][] outputWeights, double[] outputLayer, double[] trainingInputs) {
        //Compute hidden layer
        for(int j = 0; j < numHiddenNodes; j++) {
            double activation = hiddenLayerBias[j];

            for (int k = 0; k < numInputs; k++) {
                activation += trainingInputs[k] * hiddenWeights[k][j];
            }

            hiddenLayer[j] = sigmoid(activation);
        }

        //Compute output layer
        for(int j = 0; j < numOutputs; j++) {
            double activation = outputLayerBias[j];

            for (int k = 0; k < numHiddenNodes; k++) {
                activation += hiddenLayer[k] * outputWeights[k][j];
            }

            outputLayer[j] = sigmoid(activation);
        }
    }

    public static void train(){

        double learningRate = 0.1f;

        double[] hiddenLayer = new double[numHiddenNodes];
        double[] outputLayer = new double[numOutputs];

        double[] hiddenLayerBias = new double[numHiddenNodes];
        double[] outputLayerBias = new double[numOutputs];

        double[][] hiddenWeights = new double[numInputs][numHiddenNodes];
        double[][] outputWeights = new double[numHiddenNodes][numOutputs];

        //Trainings data
        double[][] trainingInputs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};

        double[][] trainingOutputs = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

        //Initialize weights
        for(int i = 0; i < numInputs; i++){
            for(int j = 0; j < numHiddenNodes; j++){
                hiddenWeights[i][j] = initWeights();
            }
        }

        for(int i = 0; i < numHiddenNodes; i++){
            for(int j = 0; j < numOutputs; j++){
                outputWeights[i][j] = initWeights();
            }
        }

        //Initialize biases
        for(int i = 0; i < numHiddenNodes; i++){
            hiddenLayerBias[i] = initWeights();
        }


        int[] trainingSetOrder = {0, 1, 2, 3};

        int numberOfEpochs = 10000;

        //Training loop
        for(int epoch = 0; epoch < numberOfEpochs; epoch++){

            shuffle(trainingSetOrder);

            for(int x = 0; x < numTrainingSets; x++){

                int i = trainingSetOrder[x];


                //Forward pass
                forwardPass(hiddenLayerBias, hiddenWeights, hiddenLayer, outputLayerBias, outputWeights, outputLayer, trainingInputs[i]);

                System.out.printf("Input: %d,%d   Output: %g\tPredicted Output: %d\n",
                                (int)trainingInputs[i][0], (int)trainingInputs[i][1], outputLayer[0], (int)trainingOutputs[i][0]);

                //Backpropagation
                backpropagation(trainingOutputs, i, outputLayer, outputWeights, hiddenLayer, outputLayerBias,
                        learningRate, hiddenLayerBias, hiddenWeights, trainingInputs);

            }
        }
    }
}