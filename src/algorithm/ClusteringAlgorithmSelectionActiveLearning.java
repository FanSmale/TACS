package algorithm;

import java.util.Arrays;

import common.*;
import exception.*;

/**
 * Clustering algorithm selection based active learning.
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

public class ClusteringAlgorithmSelectionActiveLearning extends
		ClusteringAlgorithmsBasedActiveLearning {

	/**
	 * The weight for 1NN.
	 */
	double neighorBasedWeight;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraDistanceMeasure
	 *            The given distance measure in integer.
	 * @param paraNormalizeData
	 *            Normalize data or not.
	 * @param paraAdaptiveRatio
	 *            The distance ratio for density computing.
	 * @param paraSmallBlockThreshold
	 * 		      The threshold for small blocks.
	 * @param paraInstanceSelectionStrategy
	 *            The instance selection strategy.
	 * @param paraDisorderData
	 *            Disorder data or not.
	 * @param paraQueryAmountStrategy
	 *            The query amount strategy.
	 * @param paraNeighborBasedWeight
	 *            The weight for entropy calculation.
	 ********************
	 */
	public ClusteringAlgorithmSelectionActiveLearning(String paraFilename,
			int paraDistanceMeasure, boolean paraNormalizeData,
			boolean paraDisorderData, double paraAdaptiveRatio,
			int paraSmallBlockThreshold, int paraInstanceSelectionStrategy,
			int paraQueryAmountStrategy, double paraNeighborBasedWeight) {
		super(paraFilename, paraDistanceMeasure, paraNormalizeData,
				paraDisorderData, paraAdaptiveRatio, paraSmallBlockThreshold,
				paraInstanceSelectionStrategy, paraQueryAmountStrategy);

		// System.out.println(data);
		// isQueriedArray = new boolean[data.numInstances()];

		setNeighorBasedWeight(paraNeighborBasedWeight);

		// Prepare
		// setDc(0.5);
		finalNumBlocks = 0;
	}// Of the constructor

	/**
	 ********************
	 * Reset for repeated running.
	 ********************
	 */
	public void reset() {
		super.reset();

		finalNumBlocks = 0;
		Arrays.fill(predicts, UNHANDLED);
		Arrays.fill(algorithmWinArray, 0);
	}// Of reset

	/**
	 ************************* 
	 * Set the neighbor-based weight.
	 * 
	 * @param paraWeight
	 *            The given weight.
	 ************************* 
	 */
	public void setNeighorBasedWeight(double paraWeight) {
		neighorBasedWeight = paraWeight;
	}// Of setNeighorBasedWeight

	/**
	 ************************* 
	 * Learn. The main process.
	 ************************* 
	 */
	public String learn() {
		Arrays.fill(algorithmWinArray, 0);
		finalNumBlocks = 0;

		try {
			learnBlock(wholeBlock);
		} catch (Exception ee) {
			System.out
					.println("Error occurred in ClusteringAlgorithmSelectionActiveLearning.learn()"
							+ ee);
		} // Of try
		
		//Handle the remaining instances.
		oneNnUnhandled(wholeBlock);

		String resultMessage = "";
		resultMessage += getNumQueries(); // numQueries
		resultMessage += "," + computeAccuracy(); // accuracy
		for (int i = 0; i < algorithmWinArray.length; i++) {
			resultMessage += "," + algorithmWinArray[i];
		} // Of for i
		resultMessage += "," + finalNumBlocks;

		return resultMessage;
	}// Of learn

	/**
	 ************************* 
	 * Learn. The main process.
	 * 
	 * @param paraBlock
	 *            The current block.
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 */
	public void learnBlock(int[] paraBlock) throws LabelUsedUpException,
			DuplicateQueryException {
		// Step 1. Select instances to label.
		if (paraBlock.length <= smallBlockThreshold) {
			//Do not handle them now.
			return;
		} // Of if

		// Step 1. Select instances to label.
		// System.out.println("Handling a block with " + paraBlock.length + "
		// instances.\r\n");
		int tempNumInstancesToLabel = (int) Math.sqrt(paraBlock.length);
		selectCriticalAndLabel(paraBlock, tempNumInstancesToLabel);

		// Step 2. Classify if pure
		boolean tempPure = classifyIfPure(paraBlock);
		if (tempPure) {
			return;
		} // Of if

		// Step 3. Pre-clustering and obtain the best partition.
		// Maybe more than one best algorithms
		int[] tempBestAlgorithmArray = new int[NUM_ALGORITHMS];
		int tempNumBestAlgorithms = 0;

		int[][] tempBestBlocks = null;
		double tempBestEntropy = Double.MAX_VALUE;
		int[][] tempNewBlocks = null;
		double tempEntropy;

		for (int i = 0; i < NUM_ALGORITHMS; i++) {
			// Is this algorithm available?
			if (!availableAlgorithms[i]) {
				continue;
			} // Of if

			try {
				tempNewBlocks = clusterInTwo(paraBlock, i);
			} catch (UnableToClusterInKException ee) {
				continue;
			} // Of try

			if (tempNewBlocks == null) {
				// Some algorithm may fail on some blocks.
				continue;
			} // Of if

			tempEntropy = computeWeightedEntropy(tempNewBlocks);

			System.out
					.println("Algorithm " + i + " entropy: " + tempEntropy
							+ " for " + paraBlock.length
							+ " instances splitted to "
							+ tempNewBlocks[0].length + " + "
							+ tempNewBlocks[1].length);
			if (tempBestEntropy > tempEntropy) {
				tempBestEntropy = tempEntropy;
				tempNumBestAlgorithms = 0;
				tempBestAlgorithmArray[tempNumBestAlgorithms] = i;
				tempNumBestAlgorithms++;
				tempBestAlgorithmArray[tempNumBestAlgorithms] = i;
				// tempBestAlgorithm = i;
				tempBestBlocks = tempNewBlocks;
			} else if (Math.abs(tempBestEntropy - tempEntropy) < 1e-6) {
				tempBestAlgorithmArray[tempNumBestAlgorithms] = i;
				tempNumBestAlgorithms++;
			} // Of if
		} // Of for i

		// System.out.println("ClusteringAlgorithmSelectionActiveLearning.learnBlock()
		// test 4");
		if (tempNumBestAlgorithms == 0) {
			System.out.println("No algorithm can handle this block: "
					+ Arrays.toString(paraBlock));
			System.exit(0);
		} // Of if

		for (int i = 0; i < tempNumBestAlgorithms; i++) {
			algorithmWinArray[tempBestAlgorithmArray[i]]++;
		} // Of for i

		// Step 4. Learn these two blocks.
		learnBlock(tempBestBlocks[0]);
		learnBlock(tempBestBlocks[1]);
	}// Of learnBlock

	/**
	 ************************* 
	 * Get the instance close to the center of the given set.
	 * 
	 * @param paraSet
	 *            The indices of the set.
	 * @return The instance.
	 ************************* 
	 */
	public int getCloseCenterInstance(int[] paraSet) {
		int tempCount = paraSet.length;

		// Step 1. Compute the real (virtual) center.
		double[] tempRealCenter = new double[numConditions];
		for (int i = 0; i < paraSet.length; i++) {
			for (int j = 0; j < tempRealCenter.length; j++) {
				tempRealCenter[j] += data.instance(paraSet[i]).value(j)
						/ tempCount;
			} // Of for j
		} // Of for i

		// Step 2. Get the instance most close to the virtual center.
		double tempMinimalDistance = Double.MAX_VALUE;
		double tempDistance, tempDifference;
		int tempClosestIndex = -1;
		for (int i = 0; i < paraSet.length; i++) {
			tempDistance = 0;
			for (int j = 0; j < tempRealCenter.length; j++) {
				tempDifference = tempRealCenter[j]
						- data.instance(paraSet[i]).value(j);
				tempDistance += tempDifference * tempDifference;
			} // Of for j

			// System.out.println("Distance from " + paraSet[i] + ": " +
			// tempDistance);
			if (tempDistance < tempMinimalDistance) {
				tempMinimalDistance = tempDistance;
				tempClosestIndex = paraSet[i];
			} // Of if
		} // Of for i

		return tempClosestIndex;
	}// Of getCloseCenterInstance

	/**
	 ************************* 
	 * Test the getCloseCenterInstance() method.
	 ************************* 
	 */
	public void testGetCloseCenterInstance() {
		int[] tempIndices = { 1, 3, 5, 66 };
		int tempCenterIndex = getCloseCenterInstance(tempIndices);

		System.out.println("The data are:");
		for (int i = 0; i < tempIndices.length; i++) {
			System.out.println();
			System.out.println(data.instance(tempIndices[i]));
		} // Of for i

		System.out.println("The center is: " + tempCenterIndex);
	}// Of testGetCloseCenterInstance

	/**
	 ************************* 
	 * Compute the weighted entropy of the blocks. If one block is empty, the
	 * entropy of the block will be numClasses/2. Unknown labels are predicted
	 * using 1NN within the block.
	 * 
	 * @param paraBlocks
	 *            The given blocks.
	 * @return The weighted entropy.
	 ************************* 
	 */
	public double computeWeightedEntropy(int[][] paraBlocks) {
		// Step 0. Handle the situation when weight is 0.
		if (neighorBasedWeight < 1e-6) {
			return computeEntropy(paraBlocks);
		} // Of if

		double tempNumInstances = 0;
		double resultEntropy = 0;

		for (int i = 0; i < paraBlocks.length; i++) {
			if (paraBlocks[i].length == 0) {
				// The block is not split at all.
				return data.numClasses() / 2;
			} // Of if
			tempNumInstances += paraBlocks[i].length;
		} // Of for i

		double tempEntropy;
		for (int i = 0; i < paraBlocks.length; i++) {
			tempEntropy = computeWeightedEntropy(paraBlocks[i]);
			resultEntropy += tempEntropy * paraBlocks[i].length
					/ tempNumInstances;
		} // Of for i

		return resultEntropy;
	}// Of computeWeightedEntropy

	/**
	 ************************* 
	 * Compute the weighted entropy of the block. Unknown labels are predicted
	 * using 1NN.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return The weighted entropy.
	 ************************* 
	 */
	public double computeWeightedEntropy(int[] paraBlock) {
		// Step 0. Handle the situation when weight is 0.
		if (neighorBasedWeight < 1e-6) {
			return computeEntropy(paraBlock);
		} // Of if

		// Step 1. Which instances are already queried in this block?
		int tempNumQueries = 0;
		int[] tempQueriedInstances;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempNumQueries++;
			} // Of if
		} // Of for i

		// Step 2. Handle the situation where no instance is queried.
		if (tempNumQueries == 0) {
			// No instance queried in this block.
			return data.classAttribute().numValues() / 2;
		} // Of if

		tempQueriedInstances = new int[tempNumQueries];
		int tempCounter = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempQueriedInstances[tempCounter] = paraBlock[i];
				tempCounter++;
			} // Of if
		} // Of for i

		// Step 3. First scan for queried labels
		double[] tempQueryDistribution = new double[numClasses];
		for (int i = 0; i < tempQueriedInstances.length; i++) {
			tempQueryDistribution[predicts[tempQueriedInstances[i]]]++;
		} // Of for i

		// If there is only one type of label, the entropy is 0
		int tempNonZeroCounts = 0;
		for (int i = 0; i < tempQueryDistribution.length; i++) {
			if (tempQueryDistribution[i] > 0.1) {
				tempNonZeroCounts++;
			} // Of if
		} // Of for i
		if (tempNonZeroCounts == 1) {
			// System.out.println("" + paraBlock.length + " instances, the
			// queried instance distribution is: "
			// + Arrays.toString(tempQueryDistribution) + ", return 0.");
			return 0;
		} // Of if

		// Step 4. Second scan for 1NN predicted labels.
		double[] tempPredictionDistribution = new double[numClasses];
		double tempDistance, tempMinimalDistance;
		int tempClosestNeighbor;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				continue;
			} // Of if

			tempMinimalDistance = Double.MAX_VALUE;
			tempClosestNeighbor = -1;
			for (int j = 0; j < tempNumQueries; j++) {
				tempDistance = distanceMeasure.distance(paraBlock[i],
						tempQueriedInstances[j]);
				if (tempDistance < tempMinimalDistance) {
					tempMinimalDistance = tempDistance;
					tempClosestNeighbor = tempQueriedInstances[j];
				} // Of if
			} // Of for j
			tempPredictionDistribution[predicts[tempClosestNeighbor]]++;
		} // Of for i

		// SimpleTools.variableTrackingOutput
		// System.out.println("" + paraBlock.length + " instances, the queried
		// instance distribution is: "
		// + Arrays.toString(tempQueryDistribution));
		// System.out.println("The predicted instance distribution is: " +
		// Arrays.toString(tempPredictionDistribution));

		// Step 5. Compute the weighted entropy.
		double tempWeightedTotal = tempNumQueries + neighorBasedWeight
				* (paraBlock.length - tempNumQueries);
		// System.out.println("tempWeightedTotal = " + tempWeightedTotal);
		double tempValue = 0;
		double tempEntropy = 0;
		for (int i = 0; i < numClasses; i++) {
			tempValue = (tempQueryDistribution[i] + tempPredictionDistribution[i]
					* neighorBasedWeight)
					/ tempWeightedTotal;
			// System.out.println("tempValue = " + tempValue);
			if (tempValue < 1e-6) {
				continue;
			} // Of if

			tempEntropy -= tempValue * Math.log(tempValue) / Math.log(2.0);
		} // Of for i

		// System.out.println(
		// "computeWeightedEntropy() with " + paraBlock.length + " instances,
		// tempEntropy = " + tempEntropy);
		return tempEntropy;
	}// Of computeWeightedEntropy

	/**
	 ************************* 
	 * Compute the entropy of the blocks considering queried instances only. If
	 * one block is empty, the entropy of the block will be numClasses/2.
	 * 
	 * @param paraBlocks
	 *            The given blocks.
	 * @return The entropy.
	 ************************* 
	 */
	public double computeEntropy(int[][] paraBlocks) {
		double tempNumInstances = 0;
		double resultEntropy = 0;

		for (int i = 0; i < paraBlocks.length; i++) {
			if (paraBlocks[i].length == 0) {
				// The block is not split at all.
				return data.numClasses() / 2;
			} // Of if
			tempNumInstances += paraBlocks[i].length;
		} // Of for i

		double tempEntropy;
		for (int i = 0; i < paraBlocks.length; i++) {
			tempEntropy = computeEntropy(paraBlocks[i]);
			resultEntropy += tempEntropy * paraBlocks[i].length
					/ tempNumInstances;
		} // Of for i

		return resultEntropy;
	}// Of computeEntropy

	/**
	 ************************* 
	 * Compute the entropy of the block. Only consider labeled instances.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return The entropy.
	 ************************* 
	 */
	public double computeEntropy(int[] paraBlock) {
		// Step 1. Which instances are already queried in this block?
		int tempNumQueries = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempNumQueries++;
			} // Of if
		} // Of for i

		if (tempNumQueries == 0) {
			// No instance queried in this block.
			return data.numClasses() / 2;
		} // Of if

		// Step 1. Statistics on queried labels.
		double[] tempQueryDistribution = new double[numClasses];
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempQueryDistribution[predicts[paraBlock[i]]]++;
			} // Of if
		} // Of for i

		SimpleTools.variableTrackingOutput("" + paraBlock.length
				+ " instances, the queried instance distribution is: "
				+ Arrays.toString(tempQueryDistribution));

		// Step 2. Compute the entropy.
		double tempEntropy = 0;
		double tempValue = 0;
		for (int i = 0; i < numClasses; i++) {
			tempValue = tempQueryDistribution[i] / tempNumQueries;
			if (tempValue < 1e-6) {
				continue;
			} // Of if

			tempEntropy -= tempValue * Math.log(tempValue) / Math.log(2.0);
		} // Of for i

		SimpleTools.processTrackingOutput("computeEntropy() with "
				+ paraBlock.length + " instances, tempEntropy = " + tempEntropy
				+ ", distribution = " + Arrays.toString(tempQueryDistribution));
		return tempEntropy;
	}// Of computeEntropy

	/**
	 ************************* 
	 * The main entrance
	 * 
	 * @author Fan Min
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		System.out.println("Hello.");
		// String tempFilename = "src/data/iris.arff";
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "E:/workplace/Coser2.10.1/data/wdbc.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			System.out.println("The filename is: " + tempFilename);
		} // Of if

		ClusteringAlgorithmSelectionActiveLearning tempCeal = new ClusteringAlgorithmSelectionActiveLearning(
				tempFilename, DistanceMeasure.EUCLIDEAN, true, false, 0.03, 10,
				DP_REPRESENTATIVE, 0, 0.7);

		boolean[] tempAlgorithms = new boolean[NUM_ALGORITHMS];
		Arrays.fill(tempAlgorithms, true);
		tempCeal.setAvailableAlgorithms(tempAlgorithms);
		tempCeal.reset();
		// Ceal tempCeal = new Ceal(tempFilename, DistanceMeasure.MAHALANOBIS,
		// true, 0.3);
		// Ceal tempCeal = new Ceal(tempFilename, DistanceMeasure.COSINE);

		// Ceal tempCeal = new Ceal("src/data/iris.arff",
		// DistanceMeasure.MANHATTAN);
		// Ceal tempCeal = new
		// Ceal("E:/workplace/grale/bin/data/mushroom.arff");

		// tempCeal.testGetCloseCenterInstance();
		// tempCeal.testComputeBlockWeightedEntropy();
		// testDoubleMatricesEqual();

		// tempCeal.testClusterInTwoKMeans();

		// tempCeal.testComputeDensity();
		// tempCeal.testClusterInTwoDensityPeaks();

		// tempCeal.testComputePriority();

		String resultString = tempCeal.learn();
		System.out.println(resultString);
	}// Of main
}// Of class ClusteringAlgorithmSelectionActiveLearning
