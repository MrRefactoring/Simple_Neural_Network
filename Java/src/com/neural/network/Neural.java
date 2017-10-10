package com.neural.network;

import com.google.gson.Gson;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class Neural {

    private final double colorPart = 0.75d;
    private final double frameMultiplier = 1.5d;

    private final String trained_data = "src/com/neural/newnetwork/data/trained_data.json";
    private final String MNIST_data = "src/com/neural/newnetwork/data/MNIST_data/";

    private int width;  // Сюда будем сохранять ширину картинки
    private int height;  // Сюда будем сохранять высоту картинки
    private double[][] coefficients;

    public Neural(){
        coefficients = new double[10][784];
    }

    public int analyze(String imgDir){
        int[] image = preprocessor(imgDir);
        double[] results = new double[10];
        for (int i = 0; i < results.length; i++){
            results[i] = Linear.dotAndSum(coefficients[i], image);
        }
        return Linear.argMax(results);
    }

    public int analyze(double[] image){
        double[] results = new double[10];
        for (int i = 0; i < results.length; i++){
            results[i] = Linear.dotAndSum(coefficients[i], image);
        }
        return Linear.argMax(results);
    }

    public void train(boolean retrain){
        if (!retrain && Files.exists(Paths.get(trained_data))){
            try {
                coefficients = new Gson().fromJson(new FileReader(new File(trained_data)), double[][].class);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        } else {
            double[][] images = normalize(MnistReader.getImages(MNIST_data + "train_images"));
            int[] labels = MnistReader.getLabels(MNIST_data + "train_labels");
            for (int i = 0; i < labels.length; i++){
                double[] image = images[i];
                int label = labels[i];
                int prediction = analyze(image);
                if (prediction != label){
                    coefficients[prediction] = Linear.minus(coefficients[prediction], image);
                    coefficients[label] = Linear.plus(coefficients[label], image);
                }
            }
            try{
                FileWriter writer = new FileWriter(trained_data);
                writer.write(new Gson().toJson(new Gson().toJsonTree(coefficients)));
                writer.close();
            } catch (IOException e){
                e.printStackTrace();
            }
        }
    }

    private int[] preprocessor(String imgDir){
        int[] result = null;
        try{
            BufferedImage image = ImageIO.read(new File(imgDir));
            result = toSquare(blackWhiteFilter(grayScale(image), imgDir.substring(imgDir.length() - 3).toLowerCase().equals("jpg")));
        } catch (IOException e){
            e.printStackTrace();
        }
        return result;
    }

    private int[] toSquare(int[] image) {
        int[][] cropped = crop(image);
        int max_size_side = (int) (Math.max(cropped.length, cropped[0].length) * frameMultiplier);
        int vertical_indent = (max_size_side - cropped.length) / 2;
        int horizontal_indent = (max_size_side - cropped[0].length) / 2;
        int[][] new_image = new int[max_size_side][max_size_side];
        for (int i = 0; i < cropped.length; i++){
            for (int j = 0; j < cropped[0].length; j++){
                new_image[i + vertical_indent][j + horizontal_indent] = cropped[i][j];
            }
        }
        width = max_size_side;
        height = max_size_side;
        return resize(Linear.ravel(new_image), 28, 28);
    }

    private int[] resize(int[] pixels, int width, int height) {
        BufferedImage image = new BufferedImage(this.width, this.height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < this.height; y++){
            for (int x = 0; x < this.width; x++){
                image.setRGB(x, y, pixels[y * this.width + x]);
            }
        }

        BufferedImage scaledBI = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = scaledBI.createGraphics();
        g.drawImage(image, 0, 0, width, height, null);
        g.dispose();

        int[] result = new int[width * height];
        for (int i = 0; i < height; i++){
            for (int j = 0; j < width; j++){
                result[i * width + j] = scaledBI.getRGB(j, i) & 0xff;
            }
        }
        return result;
    }

    private int[][] crop(int[] image) {
        int[] coordinates = new int[]{0, 0, 0, 0};  // y1, x1, y2, x2
        for (int i = 0; i < image.length; i += width){
            for (int j = i; j < i + width; j++){
                coordinates[0] = Math.min(coordinates[0], i / width);
                coordinates[1] = Math.min(coordinates[1], j - i);
                coordinates[2] = Math.max(coordinates[2], i / width);
                coordinates[3] = Math.max(coordinates[3], j - i);
            }
        }
        return cutter(reshape(image, width, height), coordinates);
    }

    private int[][] cutter(int[][] image, int[] coordinates){
        int[][] result = new int[coordinates[2] - coordinates[0] + 1][coordinates[3] - coordinates[1] + 1];
        for (int i = 0; i < image.length; i++){
            for (int j = 0; j < image[i].length; j++){
                if (i <= coordinates[2] && i >= coordinates[0]  && j <= coordinates[3] && j >= coordinates[1]){
                    result[i - coordinates[0]][j - coordinates[1]] = image[i][j];
                }
            }
        }
        return result;
    }

    private int[][] reshape(int[] array, int width, int height){
        int[][] result = new int[height][width];
        for (int i = 0; i < array.length / width; i++){
            for (int j = 0; j < width; j++){
                result[i][j] = array[i * width + j];
            }
        }
        return result;
    }

    private int[] blackWhiteFilter(int[] array, boolean jpg){
        int avgColor = avgColor(array);
        int[] result = new int[array.length];
        for (int i = 0; i < array.length; i++){
            if (array[i] >= avgColor){
                result[i] = jpg ? 0: 1;
            } else {
                result[i] = jpg ? 1: 0;
            }
        }
        return result;
    }

    private int avgColor(int[] array) {
        return (int) (Linear.sum(array) / array.length * colorPart) + 1;
    }

    private int[] grayScale(BufferedImage image){
        width = image.getWidth();
        height = image.getHeight();
        int[] result = new int[width * height];
        for (int y = 0; y < height; y++){
            for (int x = 0; x < width; x++){
                int pixel = image.getRGB(x, y);
                int r = (pixel >> 16) & 0xff;
                int g = (pixel >> 8) & 0xff;
                int b = pixel & 0xff;
                result[y * width + x] = (int) Math.ceil((r + g + b) / 3);
            }
        }
        return result;
    }

    private double[][] normalize(List<int[][]> images){
        double[][] result = new double[images.size()][images.get(0).length * images.get(0)[0].length];
        for (int i = 0; i < images.size(); i++){
            result[i] = Linear.div(Linear.ravel(images.get(i)), 255d);
        }
        return result;
    }

    public static void main(String[] args) {
        Neural neural = new Neural();
        neural.train(false);
        List<String> samples = List.of("eleven.jpg", "four.png", "one_1.png", "one_2.png",
                "seven_1.png", "seven_2.jpg", "six_1.png", "six_2.jpg", "six_3.jpg");
        for (String sample: samples){
            //if (sample.substring(sample.length() - 3).equals("jpg")) continue;
            System.out.println(sample);
            System.out.println(neural.analyze("src/com/neural/newnetwork/samples/" + sample));
            //if (sample.equals("one_1.png")) System.exit(0);
        }
    }
}
