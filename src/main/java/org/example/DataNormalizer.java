package org.example;

import java.util.List;

public class DataNormalizer {
    public static void normalizeData(List<double[]> dataList) {
        int numFeatures = dataList.get(0).length - 1; // Number of features in each data row

        // Calculate mean and standard deviation for each feature
        double[] mean = new double[numFeatures];
        double[] stdDev = new double[numFeatures];

        for (int i = 0; i < numFeatures; i++) {
            double sum = 0;
            for (double[] dataRow : dataList) {
                sum += dataRow[i];
            }
            mean[i] = sum / dataList.size();

            double sumSquaredDiff = 0;
            for (double[] dataRow : dataList) {
                sumSquaredDiff += Math.pow(dataRow[i] - mean[i], 2);
            }
            stdDev[i] = Math.sqrt(sumSquaredDiff / dataList.size());
        }

        // Normalize each feature in the data
        for (double[] dataRow : dataList) {
            for (int i = 0; i < numFeatures; i++) {
                dataRow[i] = (dataRow[i] - mean[i]) / stdDev[i];
            }
        }
    }
}
