package clustering;

import java.io.FileReader;
import java.util.Arrays;

import common.*;
import exception.*;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The superclass of any clustering algorithm. It is able to compute the
 * accuracy of the clustering algorithm with the external class information. It
 * is abstract since clusterInK is not implemented.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Almost finished, further revision is possible.<br>
 * Written time: July 20, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public abstract class Clustering {
	/**
	 * The number of pairs for calculating the longest.
	 */
	public static final int TIMES_FOR_FARTHEST_PAIR = 10;

	/**
	 * The balancing threshold.
	 */
	public static final double FINE_BALANCE_THRESHOLD = 0.2;

	/**
	 * The data. It should not be modified in this class.
	 */
	Instances data;

	/**
	 * The number of classes. For binary classification it is 2.
	 */
	int numClasses;

	/**
	 * The number conditional attributes.
	 */
	int numConditions;

	/**
	 * The number instances.
	 */
	int numInstances;

	/**
	 * The distance measure.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 * The whole block.
	 */
	int[] wholeBlock;

	/**
	 * The clusters.
	 */
	int[][] clusters;

	/**
	 * Data header, no instances.
	 */
	Instances dataHeader;

	/**
	 * Balance two blocks or not.
	 * 
	 * @see clusterInTwo(int[], int[])
	 */
	boolean balanceTwoBlocks = true;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure in integer.
	 ********************
	 */
	public Clustering(String paraFilename, int paraDistanceMeasure) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n"
					+ ee);
			System.exit(0);
		} // Of try
		data.setClassIndex(data.numAttributes() - 1);

		distanceMeasure = new DistanceMeasure(data, paraDistanceMeasure);

		initialize();
	}// Of the constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData
	 *            The data set.
	 * @param paraDistanceMeasure
	 *            The distance measure.
	 ********************
	 */
	public Clustering(Instances paraData, DistanceMeasure paraDistanceMeasure) {
		data = paraData;
		distanceMeasure = paraDistanceMeasure;

		initialize();
	}// Of the constructor

	/**
	 ********************
	 * Initialize.
	 ********************
	 */
	private void initialize() {
		numInstances = data.numInstances();
		numConditions = data.numAttributes() - 1;
		numClasses = data.attribute(numConditions).numValues();

		wholeBlock = new int[numInstances];
		for (int i = 0; i < numInstances; i++) {
			wholeBlock[i] = i;
		} // Of for i

		dataHeader = new Instances(data);
		dataHeader.delete();
	}// Of initialize

	/**
	 ************************* 
	 * Get semi-maximal distance of a block.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return The distance.
	 ************************* 
	 */
	public double getSemiMaximalDistance(int[] paraBlock) {
		double resultMaxDistance = -1;
		double tempDistance;
		int tempLength = paraBlock.length;
		int tempPairs = TIMES_FOR_FARTHEST_PAIR * tempLength;
		int tempFirst, tempSecond;
		for (int i = 0; i < tempPairs; i++) {
			tempFirst = (int) (Common.random.nextDouble() * tempLength);
			tempSecond = (int) (Common.random.nextDouble() * tempLength);

			tempDistance = distanceMeasure.distance(paraBlock[tempFirst],
					paraBlock[tempSecond]);
			if (resultMaxDistance < tempDistance) {
				resultMaxDistance = tempDistance;
			} // Of if
		} // Of for i
		return resultMaxDistance;
	}// Of getSemiMaximalDistance

	/**
	 ************************* 
	 * Get a pair of instances with semi-maximal distance in a block.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return The point pair.
	 ************************* 
	 */
	public double[][] getSemiMaximalDistancePair(int[] paraBlock) {
		double resultMaxDistance = -1;
		double tempDistance;
		int tempLength = paraBlock.length;
		int tempPairs = TIMES_FOR_FARTHEST_PAIR * tempLength;

		int tempFirst, tempSecond;
		int tempBestFirst = -1;
		int tempBestSecond = -1;
		for (int i = 0; i < tempPairs; i++) {
			tempFirst = (int) (Common.random.nextDouble() * tempLength);
			tempSecond = (int) (Common.random.nextDouble() * tempLength);

			tempDistance = distanceMeasure.distance(paraBlock[tempFirst],
					paraBlock[tempSecond]);
			if (resultMaxDistance < tempDistance) {
				resultMaxDistance = tempDistance;
				tempBestFirst = tempFirst;
				tempBestSecond = tempSecond;
			} // Of if
		} // Of for i

		double[][] resultPair = new double[2][data.numAttributes() - 1];
		for (int i = 0; i < resultPair[0].length; i++) {
			resultPair[0][i] = data.instance(paraBlock[tempBestFirst]).value(i);
			resultPair[1][i] = data.instance(paraBlock[tempBestSecond])
					.value(i);
		} // Of for i

		return resultPair;
	}// Of getSemiMaximalDistance

	/**
	 ****************** 
	 * Block information (e.g., with 0, 1, 2) to blocks (e.g., 3 blocks).
	 * 
	 * @param paraBlockInformation
	 *            The block information.
	 * @param paraK
	 *            The number of blocks.
	 * @return The blocks.
	 * @throws LessBlocksThanRequiredException
	 *             If there is less blocks.
	 ****************** 
	 */
	public int[][] blockInformationToBlocks(int[] paraBlockInformation,
			int paraK) throws LessBlocksThanRequiredException {
		return blockInformationToBlocks(wholeBlock, paraBlockInformation, paraK);
	}// Of blockInformationToBlocks

	/**
	 ****************** 
	 * Block information (e.g., with 0, 1, 2) to k blocks. If there were more
	 * than paraK blocks, the top (paraK - 1) blocks will be the same, while the
	 * last block contains other instances.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraBlockInformation
	 *            The block partition information
	 * @param paraK
	 *            The final number of blocks.
	 * @return The blocks.
	 * @throws LessBlocksThanRequiredException
	 *             If no enough blocks.
	 ****************** 
	 */
	public static int[][] blockInformationToBlocks(int[] paraBlock,
			int[] paraBlockInformation, int paraK)
			throws LessBlocksThanRequiredException {
		SimpleTools.variableTrackingOutput("The paraBlock has "
				+ paraBlock.length + " instances and the paraBlockInformation"
				+ " has length " + paraBlockInformation.length);
		// Step 1. Scan to see the maximal cluster number.
		int tempOriginalClusters = 0;
		for (int i = 0; i < paraBlockInformation.length; i++) {
			if (tempOriginalClusters < paraBlockInformation[i]) {
				tempOriginalClusters = paraBlockInformation[i];
			} // Of if
		} // Of for i
		tempOriginalClusters++;
		SimpleTools.processTrackingOutput("tempOriginalClusters = "
				+ tempOriginalClusters);

		if (tempOriginalClusters < paraK) {
			throw new LessBlocksThanRequiredException("" + tempOriginalClusters
					+ " clusters tries to split in " + paraK);
		} // Of if

		// Step 2. Count number of instances in each cluster.
		int[] tempCounters = new int[tempOriginalClusters];
		for (int i = 0; i < paraBlockInformation.length; i++) {
			tempCounters[paraBlockInformation[i]]++;
		} // Of for i

		for (int i = 0; i < paraK; i++) {
			if (tempCounters[i] == 0) {
				throw new LessBlocksThanRequiredException("Tries to split in "
						+ paraK + ", the cluster for " + i + " is empty.");
			} // Of if
		} // Of for i

		// Step 3. The top (paraK - 1) blocks.
		int[] tempTopSizes = new int[paraK + 1];
		int[] tempTopIndices = new int[paraK + 1];
		Arrays.fill(tempTopSizes, -1);
		tempTopSizes[0] = Integer.MAX_VALUE;
		for (int i = 0; i < tempCounters.length; i++) {
			for (int j = paraK - 1;; j--) {
				if (tempTopSizes[j] < tempCounters[i]) {
					tempTopSizes[j + 1] = tempTopSizes[j];
					tempTopIndices[j + 1] = tempTopIndices[j];
				} else {
					// Insert here.
					tempTopSizes[j + 1] = tempCounters[i];
					tempTopIndices[j + 1] = i;
					break;
				} // Of if
			} // Of for j
		} // Of for i

		// System.out.println("tempTopSizes = " +
		// Arrays.toString(tempTopSizes));
		// System.out.println("tempTopIndices = " +
		// Arrays.toString(tempTopIndices));

		// Step 4. Construct result blocks.
		int[][] resultBlocks = new int[paraK][];
		int tempRemaining = paraBlock.length;
		for (int i = 0; i < paraK - 1; i++) {
			resultBlocks[i] = new int[tempTopSizes[i + 1]];
			tempRemaining -= tempTopSizes[i + 1];
		} // Of for i
		resultBlocks[paraK - 1] = new int[tempRemaining];

		tempCounters = new int[paraK];
		boolean tempFound;
		for (int i = 0; i < paraBlockInformation.length; i++) {
			tempFound = false;
			for (int j = 1; j < paraK; j++) {
				if (paraBlockInformation[i] == tempTopIndices[j]) {
					resultBlocks[j - 1][tempCounters[j - 1]] = paraBlock[i];
					tempCounters[j - 1]++;
					tempFound = true;
					break;
				} // Of if
			} // Of for j

			// To the last block.
			if (!tempFound) {
				resultBlocks[paraK - 1][tempCounters[paraK - 1]] = paraBlock[i];
				tempCounters[paraK - 1]++;
			} // Of if

		} // Of for i

		return resultBlocks;
	}// Of blockInformationToBlocks

	/**
	 ****************** 
	 * Cluster into k blocks.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The number of sub-blocks.
	 * @return The sub-blocks.
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ****************** 
	 */
	public abstract int[][] clusterInK(int[] paraBlock, int paraK)
			throws UnableToClusterInKException;

	/**
	 ****************** 
	 * Cluster into k blocks using the whole block.
	 * 
	 * @param paraK
	 *            The number of sub-blocks.
	 * @return The sub-blocks.
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ****************** 
	 */
	public int[][] clusterInK(int paraK) throws UnableToClusterInKException {
		return clusterInK(wholeBlock, paraK);
	}// Of clusterInK

	/**
	 ****************** 
	 * Cluster into 2 blocks.
	 * 
	 * @return The sub-blocks.
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ****************** 
	 */
	public int[][] clusterInTwo() throws UnableToClusterInKException {
		clusters = clusterInK(wholeBlock, 2);
		return clusters;
	}// Of clusterInTwo

	/**
	 ****************** 
	 * Cluster into 2 blocks.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return The sub-blocks.
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ****************** 
	 */
	public int[][] clusterInTwo(int[] paraBlock)
			throws UnableToClusterInKException {
		clusters = clusterInK(paraBlock, 2);
		return clusters;
	}// Of clusterInTwo

	/**
	 ************************* 
	 * Cluster the given block in two using DBScan. Attention: should rewritten
	 * in the subclasses.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraQueriedArray
	 *            The queried instance within the block.
	 * @return Two sub-blocks
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ************************* 
	 */
	public int[][] clusterInTwo(int[] paraBlock, int[] paraQueriedArray)
			throws UnableToClusterInKException {
		return clusterInTwo(paraBlock);
	}// Of clusterInTwo

	/**
	 ****************** 
	 * Get the balance factor.
	 * 
	 * @param paraBlocks
	 *            The given array with exactly 2 blocks.
	 * @return the balance factor.
	 ****************** 
	 */
	public double getBalanceFactor(int[][] paraBlocks) {
		double tempFirstSize = paraBlocks[0].length;
		double tempSecondSize = paraBlocks[1].length;
		double tempFraction = 0;
		if (tempFirstSize < tempSecondSize) {
			tempFraction = tempFirstSize / tempSecondSize;
		} else {
			tempFraction = tempSecondSize / tempFirstSize;
		} // Of if

		return tempFraction;
	}// Of getBalanceFactor

	/**
	 ****************** 
	 * Is the blocks balanced?
	 * 
	 * @param paraBlocks
	 *            The given array with exactly 2 blocks.
	 * @return True if balanced.
	 ****************** 
	 */
	public boolean isBalanced(int[][] paraBlocks) {
		double tempFraction = getBalanceFactor(paraBlocks);

		if (tempFraction < FINE_BALANCE_THRESHOLD) {
			return false;
		} // Of if

		return true;
	}// Of isBalanced

	/**
	 ****************** 
	 * Compute the accuracy of the clusters. Use external evaluator, i.e., the
	 * class information.
	 * 
	 * @return The accuracy.
	 ****************** 
	 */
	public double computeAccuracy() {
		double resultAccuracy = 0;
		double tempTotalInstances = 0;
		int[] tempCounts = new int[data.numClasses()];
		int tempLabel;
		int tempMax;
		int tempCorrect = 0;

		System.out.println("computeAccuracy() test 1");
		for (int i = 0; i < clusters.length; i++) {
			// Initialize
			Arrays.fill(tempCounts, 0);

			System.out.println("computeAccuracy() test 1.1");
			tempTotalInstances += clusters[i].length;
			for (int j = 0; j < clusters[i].length; j++) {
				tempLabel = (int) data.instance(clusters[i][j]).classValue();
				tempCounts[tempLabel]++;
			} // Of for j

			tempMax = 0;
			for (int j = 0; j < tempCounts.length; j++) {
				if (tempMax < tempCounts[j]) {
					tempMax = tempCounts[j];
				} // Of if
			} // Of for j
			System.out.println("Block size = " + clusters[i].length
					+ ", correct = " + tempMax);
			tempCorrect += tempMax;
		} // Of for i

		System.out.println("tempTotalInstances = " + tempTotalInstances);
		resultAccuracy = tempCorrect / tempTotalInstances;
		return resultAccuracy;
	}// Of computeAccuracy

	/**
	 ****************** 
	 * Compute a subset of the data.
	 * 
	 * @param paraBlock
	 *            The block of the subset.
	 * @return The the subset.
	 ****************** 
	 */
	public Instances constructSubset(int[] paraBlock) {
		Instances resultData = new Instances(dataHeader);
		Instance tempInstance;
		for (int i = 0; i < paraBlock.length; i++) {
			tempInstance = new Instance(data.instance(i));
			resultData.add(tempInstance);
		}// Of for i
		return resultData;
	}// Of constructSubset

	/**
	 ************************* 
	 * Test the ClusterInTwo method.
	 ************************* 
	 */
	public void testClusterInTwo() {
		int[] tempBlock = { 1, 3, 49, 56, 88, 89, 99, 121, 123, 133 };
		// int[] tempBlock = {1, 3, 88, 89, 99, 121, 123, 133};
		// int[] tempBlock = {1, 88, 89, 99, 123, 133};

		// int[] tempBlock = {1, 3, 49, 56, 88, 89, 99};

		SimpleTools.consoleOutput("The original data is:");
		for (int i = 0; i < tempBlock.length; i++) {
			for (int j = 0; j < numConditions; j++) {
				SimpleTools.consoleOutput(" "
						+ data.instance(tempBlock[i]).value(j));
			} // Of for j
			SimpleTools.consoleOutput("\r\n");
		} // Of for i

		int[][] tempPartition = null;
		try {
			tempPartition = clusterInTwo(tempBlock);
		} catch (UnableToClusterInKException ee) {
			System.out.println(ee);
		} // Of try
		System.out.println("With clusterInTwo, the partition is: "
				+ Arrays.deepToString(tempPartition));
	}// Of testClusterInTwo

	/**
	 ************************* 
	 * Test the ClusterInK method.
	 * 
	 * @param paraK
	 *            k.
	 ************************* 
	 */
	public void testClusterInK(int paraK) {
		// int[] tempBlock = { 1, 3, 49, 56, 88, 89, 99, 121, 123, 133 };
		// int[] tempBlock = {1, 3, 88, 89, 99, 121, 123, 133};
		// int[] tempBlock = {1, 88, 89, 99, 123, 133};
		// int[] tempBlock = {1, 3, 49, 56, 88, 89, 99};

		// int[] tempBlock = wholeBlock;
		int[][] tempPartition = null;

		try {
			tempPartition = clusterInK(paraK);
		} catch (UnableToClusterInKException ee) {
			System.out.println(ee);
			System.exit(0);
		} // Of try

		System.out.println("For the full dataset, the partition is: "
				+ Arrays.deepToString(tempPartition));

		double tempAccuracy = computeAccuracy();
		// System.out.println("For the whole dataset, the clusters are "
		// + Arrays.deepToString(clusters));
		System.out.println("The accuracy is: " + tempAccuracy);
	}// Of testClusterInK

	/**
	 ************************* 
	 * Test the blockInformationToBlocks method.
	 ************************* 
	 */
	public static void testBlockInformationToBlocks() {
		int[] tempBlock = { 1, 3, 49, 56, 88, 89, 99, 121, 123, 133 };
		int[] tempBlockInformation = { 1, 3, 4, 0, 3, 2, 2, 1, 4, 3 };

		int[][] tempBlocks = null;
		try {
			tempBlocks = blockInformationToBlocks(tempBlock,
					tempBlockInformation, 3);
		} catch (LessBlocksThanRequiredException ee) {
			System.out.println(ee);
		} // Of testBlockInformationToBlocks

		System.out.println("The final blocks are:"
				+ Arrays.deepToString(tempBlocks));
	}// Of testBlockInformationToBlocks

	/**
	 ************************* 
	 * For unit test.
	 * 
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String args[]) {
		testBlockInformationToBlocks();
	}// Of main

}// Of class Clustering
