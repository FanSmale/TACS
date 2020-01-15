package algorithm;

import java.util.Arrays;

import common.DistanceMeasure;
import common.SimpleTools;
import exception.DuplicateQueryException;
import exception.LabelUsedUpException;
import exception.UnableToClusterInKException;


/**
 * Cluster ensemble based active learning.
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

public class ClusterEnsembleActiveLearningRJ extends ClusteringAlgorithmsBasedActiveLearning {

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename                  The given file.
	 * @param paraDistanceMeasure           The given distance measure in integer.
	 * @param paraNormalizeData             Normalize data or not.
	 * @param paraAdaptiveRatio             The distance ratio for density
	 *                                      computing.
	 * @param paraSmallBlockThreshold       Small block threshold.
	 * @param paraInstanceSelectionStrategy Critical instance selection strategy.
	 * @param paraQueryAmountStrategy       Query enough instances at a time, or one
	 *                                      by one.
	 * @param paraDisorderData              Disorder data or not.
	 ********************
	 */
	public ClusterEnsembleActiveLearningRJ(String paraFilename, int paraDistanceMeasure, boolean paraNormalizeData,
			boolean paraDisorderData, double paraAdaptiveRatio, int paraSmallBlockThreshold,
			int paraInstanceSelectionStrategy, int paraQueryAmountStrategy) {
		super(paraFilename, paraDistanceMeasure, paraNormalizeData, paraDisorderData, paraAdaptiveRatio,
				paraSmallBlockThreshold, paraInstanceSelectionStrategy, paraQueryAmountStrategy);
	}// Of the constructor

	/**
	 ************************* 
	 * Learn. The most important process. This method takes advantage of a queue for
	 * unfinished blocks.
	 ************************* 
	 */
	public String learn() {
		System.out.println("entering the learn process");
		
		String resultMessage = "";
		try {
			learnBlock(wholeBlock);
		} catch (Exception ee) {
			resultMessage += ee;
		} // Of try
		
		//tag
		//Handle the remaining instances.
		oneNnUnhandled(wholeBlock);
		
		resultMessage += "Clustering algorithm index = " + currentClusteringAlgorithmIndex ;
		resultMessage += ": " + currentClusteringAlgorithm;
		resultMessage += ", queries = " + getNumQueries(); // numQueries
		resultMessage += ", accuracy = " + computeAccuracy(); // accuracy
		resultMessage += ", final number of blocks = " + finalNumBlocks;
		// resultMessage += ", number of pure blocks = " + tempNumPureBlocks;
		// resultMessage += ", pure blocks size sum = " + tempPureBlocksSizeSum;
		resultMessage += ", number of small blocks = " + numSmallBlocks;
		resultMessage += ", misclassified = " + getNumMisclassified();
		resultMessage += ", misclassified in pure blocks = " + getNumMisclassified(PURE_BLOCK_PREDICTED);
		resultMessage += ", misclassified by kNN = " + getNumMisclassified(KNN_PREDICTED);
		resultMessage += ", misclassified by default label = " + getNumMisclassified(DEFAULT_LABELED);
		resultMessage += ", numDefaultLabeled = " + getNumInstancesByStatus(DEFAULT_LABELED);
		resultMessage += ", unhandled = " + Arrays.toString(getInstancesByStatus(UNHANDLED)) + "\r\t";
		
		return resultMessage;
	}// Of learn

	/**
	 ************************* 
	 * Learn. The main process.
	 * 
	 * @param paraBlock The current block.
	 * @throws LabelUsedUpException    If labels are used up.
	 * @throws DuplicateQueryException If an instance has been queried multiple
	 *                                 times.
	 ************************* 
	 */
	public void learnBlock(int[] paraBlock) throws LabelUsedUpException, DuplicateQueryException {
		// Step 1. Select instances to label.
		// May change later!
		System.out
				.println("ClusterEnsembleActiveLearningRJ\r\nHandling a block with " + paraBlock.length + " instances.");
		System.out.println("and the block is :" + Arrays.toString(paraBlock));
		int tempNumInstancesToLabel = (int) Math.sqrt(paraBlock.length);
		//????
		selectCriticalAndLabel(paraBlock, tempNumInstancesToLabel);

		// Step 2. Classify if pure.
		boolean tempPure = true;
		int tempFirstLabel = -1;
		int tempFirstLabelIndex = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				// if (isQueried(paraBlock[i])) {
				tempFirstLabel = predicts[paraBlock[i]];
				tempFirstLabelIndex = i;
				break;
			} // Of if
		} // Of for i

		for (int i = tempFirstLabelIndex + 1; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {

				// if (isQueried(paraBlock[i])) {
				if (predicts[paraBlock[i]] != tempFirstLabel) {
					tempPure = false;
					break;
				} // Of if
			} // Of if
		} // Of for i

		if (tempPure) {
			int tempPredicted = 0;
			// Now classify
//			for (int i = 0; i < paraBlock.length; i++) {
//				if (instanceStatusArray[paraBlock[i]] == QUERIED) {
//
//					// if (!isQueried(paraBlock[i])) {
//					known[paraBlock[i]] = tempFirstLabel;
//					tempPredicted++;
//				} // Of if
//			} // Of for i
			
			//tag
			for (int i = 0; i < paraBlock.length; i++) {
				if (instanceStatusArray[paraBlock[i]] == QUERIED) {
					System.out.println("the queried is :" + i + "in the tempPure");
					continue;
				} else {
					predicts[paraBlock[i]] = tempFirstLabel;
					tempPredicted++;
				}//of else
			} // Of for i
			System.out.println("Pure block with " + paraBlock.length + " instances and " + tempPredicted
					+ " predicted as " + tempFirstLabel);
			System.out.print("Queried: ");
			for (int i = 0; i < paraBlock.length; i++) {
				if (instanceStatusArray[paraBlock[i]] == QUERIED) {

					// if (isQueried(paraBlock[i])) {
					System.out.print(paraBlock[i] + "(" + data.instance(paraBlock[i]).value(numConditions) + "), ");
				} // Of if
			} // Of for i
			System.out.println();
			System.out.println("Whole block" + Arrays.toString(paraBlock));

			finalNumBlocks++;

			return;
		} // Of if

		// Step 3. Pre-clustering and obtain the set family through intersection.
		int[][] tempCurrentIntersections = null;
		int[][] tempFinalBlocks = null;
		int[][] tempCurrentAlgorithmBlocks = null;
		boolean tempFirstAlgorithm = true;

		for (int i = 0; i < NUM_ALGORITHMS; i++) {
			// Is this algorithm available?
			if (!availableAlgorithms[i]) {
				continue;
			} // Of if

			if (tempFirstAlgorithm) {
				System.out.println("the first algorithm is :" + i);
				try {
					tempFinalBlocks = clusterInTwo(paraBlock, i);
				} catch (UnableToClusterInKException ee) {
					continue;
				} // Of try

				tempFirstAlgorithm = false;
				continue;
			} // Of if

			try {
				System.out.println("the current algorithm is :" + i);
				tempCurrentAlgorithmBlocks = clusterInTwo(paraBlock, i);
			} catch (UnableToClusterInKException ee) {
				continue;
			} // Of try

			// Try to intersect
			System.out.println("\r\n Compute intersection");
			tempCurrentIntersections = SimpleTools.binarySetFamilyMaximalIntersection(tempFinalBlocks,
					tempCurrentAlgorithmBlocks);
//			System.out.println("the remain instances [0] is : " + Arrays.toString(SimpleTools.setMinus(tempCurrentIntersections[0], tempFinalBlocks[0], tempCurrentAlgorithmBlocks[0])));
//			System.out.println("the remain instances [1] is : " + SimpleTools.setMinus(tempCurrentIntersections[1], tempFinalBlocks[1], tempCurrentAlgorithmBlocks[1]));
			System.out.println("the array before intersection is £º" + Arrays.deepToString(tempFinalBlocks) + " \rand "
					+ Arrays.deepToString(tempCurrentAlgorithmBlocks));
			System.out.println("the intersection array is :" + Arrays.deepToString(tempCurrentIntersections));
			double tempOringialSize = tempFinalBlocks[0].length + tempFinalBlocks[1].length;
			double tempNewSize = tempCurrentIntersections[0].length + tempCurrentIntersections[1].length;
			double tempFraction = tempNewSize / tempOringialSize;
			System.out.println("\r\ntempFraction = " + tempFraction);
			if (tempFraction > 0.9) {
				tempFinalBlocks = tempCurrentIntersections;
			} // Of if
			
		} // Of for i

		System.out.println("Splitting a block with " + paraBlock.length + " instances.\r\nThe new blocks are: ");
		System.out.println(Arrays.deepToString(tempFinalBlocks));

		if (tempFinalBlocks == null) {
			return;
		} // Of if

		// Step 4. Learn these two blocks.
		learnBlock(tempFinalBlocks[0]);
		learnBlock(tempFinalBlocks[1]);
	}// Of learnBlock

	
	
	 
	
	public static void main(String[] args) {
		System.out.println("Hello.");
		String tempFilename = "src/data/iris.arff";
		//String tempFilename = "src/pub/rj/paper/data/iris.arff";
		// String tempFilename = "E:/workplace/Coser2.10.1/data/wdbc.arff";
		if (args.length >= 1) {
			tempFilename = args[0];
			System.out.println("The filename is: " + tempFilename);
		} // Of if

		ClusterEnsembleActiveLearningRJ tempCeal = new ClusterEnsembleActiveLearningRJ(tempFilename, DistanceMeasure.EUCLIDEAN, true, false, 0.03, 10,
				DP_REPRESENTATIVE, 0);

		boolean[] tempAlgorithms = new boolean[NUM_ALGORITHMS];
		Arrays.fill(tempAlgorithms, true);
		tempCeal.setAvailableAlgorithms(tempAlgorithms);
		tempCeal.reset();
//		PlotData pd = new PlotData(tempFilename);
//		pd.drawScatterPlot(tempCeal.data);
		
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
		System.out.println("the konwn[] is :" + Arrays.toString(tempCeal.predicts));
	}// Of main

}// Of ClusterEnsembleActiveLearningRJ