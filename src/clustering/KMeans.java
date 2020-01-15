package clustering;

import java.util.Arrays;

import common.*;
import exception.UnableToClusterInKException;
import weka.core.Instances;

/**
 * The kMeans clustering algorithms.
 * <p>
 * Author: <b>Fan Min</b>, <b>Shi-Ming Zhang</b> minfanphd@163.com,
 * minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: The simple version finished.<br>
 * Written time: April 10, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public class KMeans extends MeansClustering {

	/**
	 ********************
	 * The constructor for independent running.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 ********************
	 */
	public KMeans(String paraFilename, int paraDistanceMeasure) {
		super(paraFilename, paraDistanceMeasure);
	}// Of the first constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData
	 *            The data set.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 ********************
	 */
	public KMeans(Instances paraData, DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure);
	}// Of the second constructor

	/**
	 ************************* 
	 * Cluster the given block in using kMeans.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraCenters
	 *            The given centers.
	 * @return Clusters
	 ************************* 
	 */
	public int[][] clusterInK(int[] paraBlock, double[][] paraCenters) throws UnableToClusterInKException {
		// Number of blocks.
		int tempK = paraCenters.length;
		int[][] tempBlocks = new int[tempK][paraBlock.length];
		int[] tempCounters = new int[tempK];
		double[][] tempCenters = paraCenters;

		// Step 2. Cluster
		double[][] tempNewCenters;
		// At most 50 rounds.
		SimpleTools.processTrackingOutput("\r\nKMeans.clusterInK(), round ");
		for (int round = 0; round < 50; round++) {
			// Step 2.1. Partition the instances according to the centers.
			SimpleTools.processTrackingOutput("" + round + ", ");
			Arrays.fill(tempCounters, 0);
			for (int i = 0; i < paraBlock.length; i++) {
				int tempClosestCenterIndex = -1;
				double tempMinimalDistance = Double.MAX_VALUE;
				double tempDistance;
				for (int j = 0; j < tempK; j++) {
					tempDistance = distanceMeasure.distance(paraBlock[i], tempCenters[j]);
					if (tempDistance < tempMinimalDistance) {
						tempMinimalDistance = tempDistance;
						tempClosestCenterIndex = j;
					} // Of if
				} // Of for j

				tempBlocks[tempClosestCenterIndex][tempCounters[tempClosestCenterIndex]] = paraBlock[i];
				tempCounters[tempClosestCenterIndex]++;
			} // Of for i

			// Step 2.2. Obtain new centers.
			tempNewCenters = new double[tempK][numConditions];
			// The first center
			double tempValue;
			for (int i = 0; i < tempK; i++) {
				for (int j = 0; j < tempCounters[i]; j++) {
					for (int k = 0; k < numConditions; k++) {
						tempValue = data.instance(tempBlocks[i][j]).value(k) / tempCounters[i];
						// System.out.println("Adding " + tempValue + " for " +
						// i + ", " + k);
						tempNewCenters[i][k] += tempValue;
					} // Of for k
				} // Of for j
			} // Of for i

			// Step 2.3. The terminate condition
			if (SimpleTools.doubleMatricesEqual(tempCenters, tempNewCenters)) {
				break;
			} // Of if

			tempCenters = tempNewCenters;
		} // Of while

		// Step 3. Compress
		clusters = new int[tempK][];

		for (int i = 0; i < tempK; i++) {
			if (tempCounters[i] == 0) {
				throw new UnableToClusterInKException("Error occurred in KMeans.clusterInK(int[], double[][]): "
						+ "Unable to cluster the following block in " + paraCenters.length + ": \r\n"
						+ Arrays.toString(paraBlock));
			} // Of if

			clusters[i] = new int[tempCounters[i]];
			for (int j = 0; j < tempCounters[i]; j++) {
				clusters[i][j] = tempBlocks[i][j];
			} // Of for j
		} // Of for i

		return clusters;
	}// Of clusterInK

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
		SimpleTools.consoleOutput("Hello, kMeans.");
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/DLA.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		KMeans tempkMeans = new KMeans(tempFilename, DistanceMeasure.EUCLIDEAN);

		// tempkMeans.testClusterInTwo();
		tempkMeans.testClusterInK(2);
	}// Of main
}// Of KMeans
