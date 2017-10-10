package com.neural.network;

public class Linear {

    public static double dotAndSum(double[] array_one, double[] array_two){
        double amount = 0d;
        for (int i = 0; i < array_one.length; i++){
            amount += array_one[i] * array_two[i];
        }
        return amount;
    }

    public static double dotAndSum(double[] array_one, int[] array_two){
        double amount = 0d;
        for (int i = 0; i < array_one.length; i++){
            amount += array_one[i] * array_two[i];
        }
        return amount;
    }

    public static int argMax(double[] array){
        int max = 0;
        for (int i = 0; i < array.length; i++){
            if (array[i] > array[max]){
                max = i;
            }
        }
        return max;
    }

    public static int argMax(int[] array){
        int max = 0;
        for (int i = 0; i < array.length; i++){
            if (array[i] > array[max]){
                max = i;
            }
        }
        return max;
    }

    public static int[] ravel(int[][] matrix){
        int[] result = new int[matrix.length * matrix[0].length];
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix[i].length; j++){
                result[i * matrix.length + j] = matrix[i][j];
            }
        }
        return result;
    }

    public static double[] div(int[] array, double divider){
        double[] result = new double[array.length];
        for (int i = 0; i < array.length; i++){
            result[i] = array[i] / divider;
        }
        return result;
    }

    public static double[] plus(double[] array_one, double[] array_two){
        double[] result = new double[array_one.length];
        for (int i = 0; i < array_one.length; i++){
            result[i] = array_one[i] + array_two[i];
        }
        return result;
    }

    public static double[] minus(double[] array_one, double[] array_two){
        double[] result = new double[array_one.length];
        for (int i = 0; i < array_one.length; i++){
            result[i] = array_one[i] - array_two[i];
        }
        return result;
    }

    public static int sum(int[] array){
        int result = 0;
        for (int element: array){
            result += element;
        }
        return result;
    }

}
