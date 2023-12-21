package org.example;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        CSVReader.loadCSV();
        List<double[]> dataList = CSVReader.dataList;

        // Normalize data
        DataNormalizer.normalizeData(dataList);

        int inputSize = 8; // Number of features in your dataset
        int hiddenSize = 4; // Number of neurons in the hidden layer
        int outputSize = 1; // Number of output neurons (1 for binary classification)

        double learningRate = 0.01;
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenSize, outputSize, learningRate);

        int epochs = 1000; // Set the number of epochs
        neuralNetwork.train(dataList, epochs);

        // Test the trained network and provide feedback
        testNetwork(neuralNetwork, dataList);
    }

    private static void testNetwork(NeuralNetwork neuralNetwork, List<double[]> dataList) {
        int correctPredictions = 0;

        for (double[] dataRow : dataList) {
            double[] input = new double[dataRow.length - 1];
            System.arraycopy(dataRow, 0, input, 0, dataRow.length - 1);

            double[] output = neuralNetwork.predict(input);
            double predictedValue = output[0];
            double actualValue = dataRow[dataRow.length - 1];

            System.out.println("Prediction: " + predictedValue + ", Actual: " + actualValue);

            // Check if the prediction is correct (you might need to adjust the threshold based on your problem)
            if ((predictedValue >= 0.5 && actualValue == 1) || (predictedValue < 0.5 && actualValue == 0)) {
                correctPredictions++;
            }
        }

        double accuracy = (double) correctPredictions / dataList.size();
        System.out.println("Accuracy: " + accuracy);
    }
}
