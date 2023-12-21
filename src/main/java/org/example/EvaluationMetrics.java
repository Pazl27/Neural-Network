package org.example;

import java.util.List;

class EvaluationMetrics {
    public static void evaluate(NeuralNetwork neuralNetwork, List<double[]> dataList) {
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (double[] dataRow : dataList) {
            double[] input = new double[dataRow.length - 1];
            System.arraycopy(dataRow, 0, input, 0, dataRow.length - 1);

            double[] output = neuralNetwork.predict(input);
            double predictedValue = output[0];
            double actualValue = dataRow[dataRow.length - 1];

            // Check if the prediction is correct
            if ((predictedValue >= 0.5 && actualValue == 1) || (predictedValue < 0.5 && actualValue == 0)) {
                if (actualValue == 1) {
                    truePositives++;
                } else {
                    trueNegatives++;
                }
            } else {
                if (actualValue == 1) {
                    falseNegatives++;
                } else {
                    falsePositives++;
                }
            }
        }

        double precision = (double) truePositives / (truePositives + falsePositives);
        double recall = (double) truePositives / (truePositives + falseNegatives);
        double f1Score = 2 * (precision * recall) / (precision + recall);
        double accuracy = (double) (truePositives + trueNegatives) / dataList.size();

        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("F1-Score: " + f1Score);
        System.out.println("Accuracy: " + accuracy);
    }
}
