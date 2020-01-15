package clustering;

import java.util.Arrays;

import common.DistanceMeasure;
import common.SimpleTools;
import exception.LessBlocksThanRequiredException;
import exception.UnableToClusterInKException;
import weka.core.Instances;

/**
 * The fuzzy c-means clustering algorithms.
 * <p>
 * Author: <b>Fan Min</b>, <b>Shi-Ming Zhang</b> minfanphd@163.com,
 * minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: The simple version finished. Kernels may be added in the future<br>
 * Written time: April 10, 2019. <br>
 * Last modify time: July 23, 2019.
 */

public class FCM extends MeansClustering {

	/**
	 * zsm?
	 */
	public static final int BNUMBER = 2;

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
	public FCM(Instances paraData, DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure);
	}// Of the constructor

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
	public FCM(String paraFilename, int paraDistanceMeasure) {
		super(paraFilename, paraDistanceMeasure);
	}// Of the constructor

	/**
	 ********************
	 * Calculate membership.
	 * 
	 * @param paraCurrentCenters
	 * @param paraBlock
	 * @return Similarity of every instance to all centers.
	 ********************
	 */
	double[][] calculateMembership(double[][] paraCurrentCenters, int[] paraBlock) {
		double[][] memberShipMatrix = new double[paraBlock.length][paraCurrentCenters.length];
		double tempDenominator = 0;
		double tempNumerator = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			// Initialize.
			tempDenominator = 0;
			tempNumerator = 0;

			// Compute the denominator for the current instance.
			for (int j = 0; j < memberShipMatrix[0].length; j++) {
				tempDenominator += Math.pow(distanceMeasure.euclideanDistance(paraBlock[i], paraCurrentCenters[j]),
						-2 / (BNUMBER - 1));
			} // Of for j

			// Compute the numerator for the current instance wrt. current
			// center.
			for (int j = 0; j < memberShipMatrix[0].length; j++) {
				tempNumerator = Math.pow(distanceMeasure.euclideanDistance(paraBlock[i], paraCurrentCenters[j]),
						-2 / (BNUMBER - 1));
				memberShipMatrix[i][j] = tempNumerator / tempDenominator;
			} // Of for j
		} // Of for i
		return memberShipMatrix;
	}// Of calculateMembership

	/**
	 *********************
	 * Jf means linear planning formula
	 * 
	 * @param paraMemberShip
	 *            The memberShip matrix
	 * @param paraBlock
	 * @param paraCenters
	 * @return next Jf
	 *********************
	 */
	double calculateJf(double[][] paraMemberShip, int[] paraBlock, double[][] paraCenters) {
		double tempJfResult = 0;
		for (int i = 0; i < paraMemberShip[0].length; i++) {
			for (int j = 0; j < paraBlock.length; j++) {
				tempJfResult += Math.pow(paraMemberShip[j][i], BNUMBER)
						* Math.pow(distanceMeasure.euclideanDistance(paraBlock[j], paraCenters[i]), 2);
			} // Of for j
		} // Of for i
		return tempJfResult;
	}// Of calculateJf

	/**
	 ********************
	 * Get the centers of the next round.
	 * 
	 * @param paraMemberShip
	 * @param paraBlock
	 * @return next center of iteration
	 ********************
	 */
	double[][] iterationCenters(double[][] paraMemberShip, int[] paraBlock) {
		double[][] NewCenters = new double[paraMemberShip[0].length][numConditions];
		double tempDenominator;

		for (int i = 0; i < paraMemberShip[0].length; i++) {
			double[] tempNumerator = new double[numConditions];
			tempDenominator = 0;
			for (int j = 0; j < paraBlock.length; j++) {
				tempDenominator += Math.pow(paraMemberShip[j][i], BNUMBER);
				for (int k = 0; k < numConditions; k++) {
					tempNumerator[k] += Math.pow(paraMemberShip[j][i], BNUMBER) * data.instance(paraBlock[j]).value(k);
				} // Of for k
			} // of for j
			for (int l = 0; l < numConditions; l++) {
				tempNumerator[l] /= tempDenominator;
			} // Of for l
			NewCenters[i] = tempNumerator;
		} // Of for i
		return NewCenters;
	}// Of iterationCenters

	/**
	 *********************
	 * FCM clusterInK.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraCenters
	 *            The orignal centers.
	 * @return result clustering.
	 * @throws UnableToClusterInKException
	 *             When the algorithm fails.
	 *********************
	 */
	public int[][] clusterInK(int[] paraBlock, double[][] paraCenters) throws UnableToClusterInKException {
		int[][] result = null;
		if (paraBlock.length == 2) {
			result = new int[2][1];
			result[0][0] = paraBlock[0];
			result[1][0] = paraBlock[1];
			return result;
		} // Of if
		double[][] tempCenters = paraCenters;

		double tempOldJfResult = Double.MAX_VALUE;
		double[][] tempMemberShip = calculateMembership(tempCenters, paraBlock);
		double tempNewJfResult = calculateJf(tempMemberShip, paraBlock, tempCenters);
		while (tempNewJfResult - tempOldJfResult > 1e-6) {
			tempOldJfResult = tempNewJfResult;
			tempCenters = iterationCenters(tempMemberShip, paraBlock);
			tempMemberShip = calculateMembership(tempCenters, paraBlock);
			tempNewJfResult = calculateJf(tempMemberShip, paraBlock, tempCenters);
		} // Of while

		int tempClassificationMark = 0;
		double tempClassificationComparison;
		int[] tempResult = new int[paraBlock.length];
		for (int i = 0; i < paraBlock.length; i++) {
			tempClassificationComparison = 0;
			for (int j = 0; j < tempMemberShip[0].length; j++) {
				if (tempClassificationComparison < tempMemberShip[i][j]) {
					tempClassificationComparison = tempMemberShip[i][j];
					tempClassificationMark = j;
				} // Of if
			} // Of for j
			tempResult[i] = tempClassificationMark;
		} // Of for i

		try {
			result = blockInformationToBlocks(paraBlock, tempResult, tempCenters.length);
			// SimpleTools.processTrackingOutput("In FCM, lengths = " +
			// result[0].length + ", " + result[1].length
			// + ", tempCenters.length = " + tempCenters.length + ",
			// tempResult[0] = " + tempResult[0] + "\r\n");
		} catch (LessBlocksThanRequiredException ee) {
			SimpleTools.processTrackingOutput(ee + "\r\n");
			throw new UnableToClusterInKException("FCM cannot cluster the block in " + tempCenters.length + ": "
					+ paraBlock.length + " (" + paraBlock[0] + " ...)");
		} // Of try

		return result;
	}// Of clusterInK

	/**
	 ************************* 
	 * Test this class.
	 * 
	 * @author Shi-Ming Zhang
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		String tempFilename = "src/data/iris.arff";

		FCM tempFCM = new FCM(tempFilename, DistanceMeasure.EUCLIDEAN);

		int[] tempIns = new int[tempFCM.numInstances];
		for (int i = 0; i < tempFCM.numInstances; i++) {
			tempIns[i] = i;
		} // Of for i
		int[][] test = null;
		try {
			test = tempFCM.clusterInTwo(tempIns);
		} catch (Exception ee) {
			System.out.println(ee);
			System.exit(0);
		} // Of try

		System.out.println(Arrays.deepToString(test));
	}// Of main

}// Of class FCM
