package org.example;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        CSVReader.loadCSV();
        List<double[]> dataList = CSVReader.dataList;

        // Normalize data
        DataNormalizer.normalizeData(dataList);

        int inputSize = 7; // Number of features in your dataset
        int hiddenSize = 7; // Number of neurons in the hidden layer
        int outputSize = 1; // Number of output neurons (1 for binary classification)

        double learningRate = 0.5; // Adjust the learning rate
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize, learningRate);

        int epochs = 100000; // Increase the number of epochs

        // Debug-Ausgaben während des Trainings
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;
            for (double[] dataRow : dataList) {
                double[] input = new double[dataRow.length - 1];
                System.arraycopy(dataRow, 0, input, 0, dataRow.length - 1);

                double[] output = neuralNetwork.predict(input);
                double target = dataRow[dataRow.length - 1];

                double error = target - output[0];
                totalError += 0.5 * Math.pow(error, 2);

                // Debug-Ausgaben
                System.out.println("Epoch: " + epoch + ", Error: " + totalError + ", Prediction: " + output[0] + ", Target: " + target);
            }
        }

        // Nach dem Training: Gewichte speichern
        neuralNetwork.saveWeights("trained_weights.txt");

        // Beim Start des Programms: Gewichte laden
        neuralNetwork.loadWeights("trained_weights.txt");

        // Test the trained network and provide feedback
        EvaluationMetrics.evaluate(neuralNetwork, dataList);
    }
}
