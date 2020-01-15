package algorithm;

import java.io.FileReader;

import weka.classifiers.trees.*;
import weka.core.Instances;

public class TestJ48 {

	/**
	 ************************* 
	 * Test this class.
	 * 
	 * @author Fan Min
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		System.out
				.println("Hello, active learning. I only want to test the constructor and kNN.");
		String tempFilename = "src/data/mushroom.arff";
		// String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/r15.arff";
		// String tempFilename = "src/data/banana.arff";
		Instances data = null;
		try {
			FileReader fileReader = new FileReader(tempFilename);
			data = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + tempFilename + "\r\n"
					+ ee);
			System.exit(0);
		} // Of try
		
		J48 tempJ48 = new J48();
		Instances trainingData = new Instances(data);
		for (int i = data.numInstances() - 1; i >= 0; i -= 2) {
			trainingData.delete(i);
		}//Of for i
		
		int tempCounter = 0;
		try {
			tempJ48.buildClassifier(trainingData);
			int tempResult;
			for (int i = 0; i < data.numInstances(); i++) {
				tempResult = (int)tempJ48.classifyInstance(data.instance(i));
				if (tempResult == data.instance(i).classValue()) {
					tempCounter ++;
				}//Of if
			}//Of for i
		} catch (Exception ee) {
			System.out.println(ee);
		}
	}//Of main
}//Of class TestJ48 
