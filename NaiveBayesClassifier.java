
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

public class NaiveBayesClassifier {
    // Stores the conditional probabilities
    private static Map<String, Map<String, Double>> probabilities;
    // Stores the class probabilities
    private static Map<String, Double> classCounts;
    private static int totalInstances;

    // Define possible values for each feature
    private static final Map<String, Set<String>> POSSIBLE_VALUES = new HashMap<>();
    static {
        POSSIBLE_VALUES.put("age", new HashSet<>(Arrays.asList("10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-99")));
        POSSIBLE_VALUES.put("menopause", new HashSet<>(Arrays.asList("lt40", "ge40", "premeno")));
        POSSIBLE_VALUES.put("tumor-size", new HashSet<>(Arrays.asList("0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59")));
        POSSIBLE_VALUES.put("inv-nodes", new HashSet<>(Arrays.asList("0-2", "3-5", "6-8", "9-11", "12-14", "15-17", "18-20", "21-23", "24-26", "27-29", "30-32", "33-35", "36-39")));
        POSSIBLE_VALUES.put("node-caps", new HashSet<>(Arrays.asList("yes", "no")));
        POSSIBLE_VALUES.put("deg-malig", new HashSet<>(Arrays.asList("1", "2", "3")));
        POSSIBLE_VALUES.put("breast", new HashSet<>(Arrays.asList("left", "right")));
        POSSIBLE_VALUES.put("breast-quad", new HashSet<>(Arrays.asList("left_up", "left_low", "right_up", "right_low", "central")));
        POSSIBLE_VALUES.put("irradiat", new HashSet<>(Arrays.asList("yes", "no")));
    }

    // Define the features and class labels
    private static final String[] FEATURES = {
        "age", "menopause", "tumor-size", "inv-nodes", "node-caps",
        "deg-malig", "breast", "breast-quad", "irradiat"
    };
    private static final String[] CLASS_LABELS = { "no-recurrence-events", "recurrence-events" };

    public static void main(String[] args) throws Exception {
        // Initialize the maps for probabilities and class counts
        probabilities = new HashMap<>();
        classCounts = new HashMap<>();

        // Check if the correct number of arguments are provided
        if (args.length != 2) {
            System.out.println("Please provide the testing and training files as a command line argument.");
            System.out.println("You provided " + args.length + " arguments.");
            return;
        }

        // File paths for the test and training data
        String testFilePath = args[0];
        String trainingFilePath = args[1];

        // Load and train the model with the training data
        List<Instance> trainingInstances = loadInstances(trainingFilePath);
        train(trainingInstances);

        // Load the test data and evaluate the model
        List<Instance> testingInstances = loadInstances(testFilePath);
        double correctCount = 0;
        for (Instance instance : testingInstances) {
            String actual = instance.getClassLabel();
            String prediction = predict(instance);

            System.out.println("Actual: " + actual + " Predicted: " + prediction);
            System.out.println();
            if (actual.equals(prediction)) {
                correctCount++;
            }
        }
        System.out.println("Accuracy: " + (double) correctCount / testingInstances.size() * 100 + "%");
    }

    // Train the Naive Bayes model
    public static void train(List<Instance> instances) {
        totalInstances = instances.size();

        // Initialize counts to 1 for Laplace smoothing
        for (String classLabel : CLASS_LABELS) {
            classCounts.put(classLabel, 1.0);
            probabilities.put(classLabel, new HashMap<>());
            for (String feature : FEATURES) {
                for (String value : POSSIBLE_VALUES.get(feature)) {
                    probabilities.get(classLabel).put(feature + "=" + value, 1.0);
                }
            }
        }

        // Count instances
        for (Instance instance : instances) {
            String classLabel = instance.getClassLabel();
            // Increment class count
            classCounts.put(classLabel, classCounts.get(classLabel) + 1);
            // Increment feature counts
            for (String feature : FEATURES) {
                String value = instance.getFeatureValue(feature);
                probabilities.get(classLabel).put(feature + "=" + value,
                        probabilities.get(classLabel).get(feature + "=" + value) + 1.0);
            }
        }

        // Calculate probabilities
        for (String classLabel : CLASS_LABELS) {
            // Calculate class probabilities
            classCounts.put(classLabel, classCounts.get(classLabel) / totalInstances);
            // Calculate feature probabilities
            for (String feature : FEATURES) {
                double total = 0;
                // Calculate value probabilities
                for (String value : POSSIBLE_VALUES.get(feature)) {
                    total += probabilities.get(classLabel).get(feature + "=" + value);
                }
                for (String value : POSSIBLE_VALUES.get(feature)) {
                    probabilities.get(classLabel).put(feature + "=" + value,
                            probabilities.get(classLabel).get(feature + "=" + value) / total);
                }
            }
        }

        // Print conditional probabilities
        for (String classLabel : CLASS_LABELS) {
            System.out.println("Class: " + classLabel);
            for (String feature : FEATURES) {
                for (String value : POSSIBLE_VALUES.get(feature)) {
                    double probability = probabilities.get(classLabel).get(feature + "=" + value);
                    System.out.println("P(" + feature + "=" + value + "|" + classLabel + ") = " + probability);
                }
            }
            System.out.println();
        }

        // Print class probabilities
        for (String classLabel : CLASS_LABELS) {
            double probability = classCounts.get(classLabel);
            System.out.println("Class Probabilities:");
            System.out.println("P(" + classLabel + ") = " + probability);
        }
        System.out.println();
    }

    // Predict the class for a given instance
    public static String predict(Instance instance) {
        Map<String, Double> scores = new HashMap<>();
        for (String classLabel : CLASS_LABELS) {
            scores.put(classLabel, classCounts.get(classLabel));

            for (String feature : FEATURES) {
                String value = instance.getFeatureValue(feature);
                scores.put(classLabel,
                        scores.get(classLabel) * probabilities.get(classLabel).get(feature + "=" + value));
            }
        }
        String predictedClass = null;
        double maxScore = Double.NEGATIVE_INFINITY;
        for (Map.Entry<String, Double> entry : scores.entrySet()) {
            if (entry.getValue() > maxScore) {
                maxScore = entry.getValue();
                predictedClass = entry.getKey();
            }
        }
        // Print scores and predicted class for the input vector
        System.out.println("Instance: " + instance.getID());
        for (String classLabel : CLASS_LABELS) {
            System.out.println("score(" + classLabel + ", X) = " + scores.get(classLabel));
        }
        System.out.println("Predicted class: " + predictedClass);
        return predictedClass;
    }

    // Load instances from a file
    public static List<Instance> loadInstances(String filePath) {
        List<Instance> instances = new ArrayList<>();
        String[] featureLabels;
        try (Scanner s = new Scanner(new File(filePath))) {
            if (s.hasNextLine()) {
                String[] firstLine = s.nextLine().split(",");
                featureLabels = Arrays.copyOfRange(firstLine, 2, firstLine.length);
                while (s.hasNextLine()) {
                    String line = s.nextLine();
                    String[] values = line.split(",");
                    Instance newInstance = new Instance(values[0], values[1]);
                    for (int i = 2; i < values.length; i++) {
                        newInstance.addFeature(featureLabels[i - 2], values[i]);
                    }
                    instances.add(newInstance);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return instances;
    }

    // Inner class representing an instance of data
    public static class Instance {
        String id;
        String instanceClass;
        Map<String, String> features;

        public Instance(String id, String instanceClass) {
            this.id = id;
            this.instanceClass = instanceClass;
            features = new HashMap<>();
        }

        void addFeature(String label, String value) {
            features.put(label, value);
        }

        String getFeatureValue(String s) {
            return features.get(s);
        }

        String getClassLabel() {
            return instanceClass;
        }

        String getID() {
            return id;
        }

        void print() {
            System.out.println("ID:" + id + " Class:" + instanceClass + "Features:");
            for (String s : features.keySet()) {
                System.out.print(s + ", ");
            }
        }
    }
}