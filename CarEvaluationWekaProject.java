import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
import weka.core.SerializationHelper;
import javax.swing.*;
import java.io.FileWriter;
import java.util.Random;

public class CarEvaluationWekaProject {

    public static void main(String[] args) throws Exception {

        // ---------------------------------------------------------------------
        //  LOAD DATA
        // ---------------------------------------------------------------------
        DataSource source = new DataSource("car.arff");
        Instances data = source.getDataSet();

        // By default, if the class index is not set, set it to the last attribute
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        // Print dataset summary
        System.out.println("=== Car Evaluation Dataset Summary ===");
        System.out.println("Number of attributes: " + data.numAttributes());
        System.out.println("Number of instances : " + data.numInstances());
        System.out.println("Class index         : " + data.classIndex());
        System.out.println();

        // ---------------------------------------------------------------------
        //  PRINT OUT THE DISTRIBUTION OF THE CLASS
        // ---------------------------------------------------------------------
        printClassDistribution(data);

        // ---------------------------------------------------------------------
        //  M1: MODEL TRAINED ON 70% / TEST ON 30%
        // ---------------------------------------------------------------------
        System.out.println("\n=== Building Model M1 (70% train / 30% test) ===");

        // (a) Shuffle the data
        data.randomize(new Random(1));

        // (b) Split the data
        int trainSize1 = (int) Math.round(data.numInstances() * 0.70);
        int testSize1 = data.numInstances() - trainSize1;
        Instances trainData1 = new Instances(data, 0, trainSize1);
        Instances testData1  = new Instances(data, trainSize1, testSize1);

        // (c) Build the Decision Tree (J48 is Weka's C4.5 implementation)
        J48 j48_1 = new J48();
        j48_1.buildClassifier(trainData1);
        Classifier M1 = j48_1;

        // (d) Evaluate M1 on the 30% test data
        Evaluation evalM1 = new Evaluation(trainData1);
        evalM1.evaluateModel(M1, testData1);

        // (e) Print accuracy and F1-score
        double accuracyM1 = evalM1.pctCorrect();
        double f1ScoreM1  = evalM1.weightedFMeasure();  // Weighted average F1
        System.out.println("M1 Accuracy (%): " + String.format("%.2f", accuracyM1));
        System.out.println("M1 F1-Score    : " + String.format("%.4f", f1ScoreM1));
        System.out.println(evalM1.toSummaryString("\n=== M1 Summary ===\n", false));
        System.out.println(evalM1.toClassDetailsString("\n=== M1 Detailed Accuracy By Class ===\n"));
        System.out.println(evalM1.toMatrixString("\n=== M1 Confusion Matrix ===\n"));

        SerializationHelper.write("M1.model", M1);
        String dotM1 = j48_1.graph();
        try (FileWriter fw = new FileWriter("M1.dot")) {
            fw.write(dotM1);
        }


        // ---------------------------------------------------------------------
        //  M2: MODEL TRAINED ON 50% / TEST ON 50%
        // ---------------------------------------------------------------------
        System.out.println("\n=== Building Model M2 (50% train / 50% test) ===");

        // (a) Shuffle again with a different seed (to ensure a different partition)
        data.randomize(new Random(2));

        // (b) Split the data
        int trainSize2 = (int) Math.round(data.numInstances() * 0.50);
        int testSize2 = data.numInstances() - trainSize2;
        Instances trainData2 = new Instances(data, 0, trainSize2);
        Instances testData2  = new Instances(data, trainSize2, testSize2);

        // (c) Build the Decision Tree
        J48 j48_2 = new J48();
        j48_2.buildClassifier(trainData2);
        Classifier M2 = j48_2;

        // (d) Evaluate M2 on its 50% test data
        Evaluation evalM2 = new Evaluation(trainData2);
        evalM2.evaluateModel(M2, testData2);

        // (e) Print accuracy and F1-score
        double accuracyM2 = evalM2.pctCorrect();
        double f1ScoreM2  = evalM2.weightedFMeasure();
        System.out.println("M2 Accuracy (%): " + String.format("%.2f", accuracyM2));
        System.out.println("M2 F1-Score    : " + String.format("%.4f", f1ScoreM2));
        System.out.println(evalM2.toSummaryString("\n=== M2 Summary ===\n", false));
        System.out.println(evalM2.toClassDetailsString("\n=== M2 Detailed Accuracy By Class ===\n"));
        System.out.println(evalM2.toMatrixString("\n=== M2 Confusion Matrix ===\n"));

        SerializationHelper.write("M2.model", M2);
        String dotM2 = j48_2.graph();
        try (FileWriter fw = new FileWriter("M2.dot")) {
            fw.write(dotM2);
        }

        // ---------------------------------------------------------------------
        //  CROSS-VALIDATION
        // ---------------------------------------------------------------------
        Evaluation crossEval = new Evaluation(data);
        crossEval.crossValidateModel(j48_1, data, 10, new Random(1));

        System.out.println("=== 10-Fold CV ===");
        System.out.println("Accuracy: " + crossEval.pctCorrect());
        System.out.println("F1-Score: " + crossEval.weightedFMeasure());


        // ---------------------------------------------------------------------
        //  COMPARE M1 VS M2 VS 10-FOLD CV
        // ---------------------------------------------------------------------
        System.out.println("\n=== Comparison of M1 vs M2 ===");
        System.out.println("M1 - 70/30 Accuracy : " + String.format("%.2f", accuracyM1) +
                " | F1: " + String.format("%.4f", f1ScoreM1));
        System.out.println("M2 - 50/50 Accuracy : " + String.format("%.2f", accuracyM2) +
                " | F1: " + String.format("%.4f", f1ScoreM2));
        System.out.println("10-Fold CV Accuracy : " + String.format("%.2f", crossEval.pctCorrect()) +
                " | F1: " + String.format("%.4f", crossEval.weightedFMeasure()));


        System.out.println("\nPossible Explanation:");
        System.out.println(" - M1 trains on more data (70%), so it might perform better if the dataset is small.");
        System.out.println(" - M2 has less training data (50%), so it might have slightly lower performance.");
        System.out.println(" - Actual results depend on random seeds, the dataset size, etc.");

        // -------------------------------------------------------------------------------------
        //  GENERATE (PLOT) THE DECISION TREE OF M1 & M2 and Print the DOT format (GraphViz).
        // -------------------------------------------------------------------------------------
        System.out.println("\n=== Decision Tree for M1 (Textual Representation) ===");
        System.out.println(j48_1.toString());
        System.out.println("\n=== Decision Tree for M1 (DOT Format) ===");
        System.out.println(j48_1.graph());

        System.out.println("\n=== Decision Tree for M2 (Textual Representation) ===");
        System.out.println(j48_2.toString());
        System.out.println("\n=== Decision Tree for M2 (DOT Format) ===");
        System.out.println(j48_2.graph());

        // Visualize the decision tree
        visualizeTree(j48_1);
        visualizeTree(j48_2);
    }

    // -------------------------------------------------------------------------
    // Print Class Distribution
    // -------------------------------------------------------------------------
    private static void printClassDistribution(Instances data) {
        System.out.println("=== Class Distribution ===");
        Attribute classAttr = data.classAttribute();
        int numInstances = data.numInstances();

        int[] classCounts = new int[classAttr.numValues()];
        for (int i = 0; i < numInstances; i++) {
            int classIndex = (int) data.instance(i).classValue();
            classCounts[classIndex]++;
        }

        for (int i = 0; i < classAttr.numValues(); i++) {
            double percentage = (classCounts[i] / (double) numInstances) * 100.0;
            System.out.printf("  %s: %d (%.2f%%)%n", classAttr.value(i), classCounts[i], percentage);
        }
        System.out.println("===========================================");
    }

    // -------------------------------------------------------------------------
    // Visualize Decision Tree
    // -------------------------------------------------------------------------
    public static void visualizeTree(J48 tree) {
        try {
            JFrame frame = new JFrame("Decision Tree Visualizer");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            TreeVisualizer tv = new TreeVisualizer(null,
                    tree.graph(),
                    new PlaceNode2());
            frame.setLayout(new java.awt.BorderLayout());
            frame.add(tv, java.awt.BorderLayout.CENTER);
            frame.setSize(800, 600);
            frame.setVisible(true);

            tv.fitToScreen();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
