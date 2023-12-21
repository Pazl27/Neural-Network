package org.example;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private double[][] weights;
    private double learningRate;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.learningRate = learningRate;
        this.weights = new double[hiddenSize][inputSize + 1]; // +1 for bias

        // Initialize weights using He initialization
        Random random = new Random();
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize + 1; j++) {
                weights[i][j] = random.nextGaussian() * Math.sqrt(2.0 / inputSize);
            }
        }
    }

    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double[] predict(double[] input) {
        double[] hiddenLayerOutput = new double[weights.length];

        // Berechne Ausgabe der versteckten Schicht
        for (int i = 0; i < weights.length; i++) {
            double sum = weights[i][0]; // Bias term
            for (int j = 1; j < weights[i].length; j++) {
                sum += weights[i][j] * input[j - 1]; // -1 because input doesn't include bias
            }
            hiddenLayerOutput[i] = sigmoid(sum);
        }

        return hiddenLayerOutput;
    }

    public void train(List<double[]> data, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (double[] dataRow : data) {
                double[] input = new double[dataRow.length - 1];
                System.arraycopy(dataRow, 0, input, 0, dataRow.length - 1);

                double[] output = predict(input);
                double error = dataRow[dataRow.length - 1] - output[0];

                // Update weights
                for (int i = 0; i < weights.length; i++) {
                    weights[i][0] += learningRate * error; // Update bias term
                    for (int j = 1; j < weights[i].length; j++) {
                        weights[i][j] += learningRate * error * input[j - 1]; // -1 because input doesn't include bias
                    }
                }
            }
        }
    }
}