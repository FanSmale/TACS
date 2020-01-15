package algorithm;

import java.util.Arrays;

import common.BlockQueue;
import common.SimpleTools;
import exception.*;

/**
 * Active learning through clustering algorithm selection. The number of labels
 * is specified by the user.
 * <p>
 * See Min F. et al. Active learning through clustering algorithm selection,
 * International Journal of Machine Learning and Cybernetics, 2020.
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

public class SpecifiedLabelsAlgorithmSelection extends ClusteringAlgorithmSelectionActiveLearning {

	/**
	 * Total labels provided by the oracle.
	 */
	int numTotalLabels;

	/**
	 * Representative labels provided by the oracle.
	 */
	int numRepresentativeLabels;

	/**
	 * The number of best algorithm for the current round.
	 */
	int numBestAlgorithms;

	/**
	 * The best algorithm for the current round.
	 */
	int[] bestAlgorithmArray;

	/**
	 * Retrospect to select the clustering technique.
	 */
	boolean retrospect;
	
	/**
	 ********************
	 * The first constructor.
	 * 
	 * @param paraFilename
	 *            The dataset filename.
	 * @param paraDistanceMeasure
	 *            The given distance measure in integer.
	 * @param paraNormalizeData
	 *            Normalize data or not.
	 * @param paraAdaptiveRatio
	 *            The distance ratio for density computing.
	 * @param paraSmallBlockThreshold
	 *            Small block threshold.
	 * @param paraInstanceSelectionStrategy
	 *            The instance selection strategy, DP_REPRESENTATIVE or
	 *            MAX_TOTAL_DISTANCE.
	 * @param paraQueryAmountStrategy
	 *            The query amount strategy.
	 * @param paraNeighborBasedWeight
	 *            The weight for unlabeled instances.
	 * @param paraTotalLabelFraction
	 *            The fraction of total labels that can be provided.
	 * @param paraRepresentativeFraction
	 *            The fraction of representative labels.
	 * @param paraDisorderData
	 *            Disorder data or not.
	 * @param paraRetrospect
	 *            Retrospect or not.
	 ********************
	 */
	public SpecifiedLabelsAlgorithmSelection(String paraFilename, int paraDistanceMeasure, boolean paraNormalizeData,
			boolean paraDisorderData, double paraAdaptiveRatio,
			int paraSmallBlockThreshold, int paraInstanceSelectionStrategy,
			int paraQueryAmountStrategy, double paraNeighborBasedWeight, double paraTotalLabelFraction,
			double paraRepresentativeFraction, boolean paraRetrospect) {
		super(paraFilename, paraDistanceMeasure, paraNormalizeData, paraDisorderData, paraAdaptiveRatio, paraSmallBlockThreshold,
				paraInstanceSelectionStrategy, paraQueryAmountStrategy, paraNeighborBasedWeight);

		numTotalLabels = (int) (data.numInstances() * paraTotalLabelFraction);
		numRepresentativeLabels = (int) (numTotalLabels * paraRepresentativeFraction);
		numRemainingQueries = numTotalLabels;
		retrospect = paraRetrospect;
	}// Of the first constructor

	/**
	 ********************
	 * Reset for repeated running.
	 ********************
	 */
	public void reset() {
		super.reset();
		numRemainingQueries = numTotalLabels;
		bestAlgorithmArray = new int[NUM_ALGORITHMS];
	}// Of reset

	/**
	 ************************* 
	 * Compute the best clusters for the current block. This method needs two
	 * more member variables: bestAlgorithmArray and numBestAlgorithms.
	 * 
	 * @param paraBlock
	 *            The current block.
	 * @return The best sub-blocks.
	 * @throws UnableToClusterInKException
	 *             When all algorithms fail.
	 ************************* 
	 */
	public int[][] computeBestSubBlocks(int[] paraBlock) throws UnableToClusterInKException {
		Arrays.fill(bestAlgorithmArray, -1);
		numBestAlgorithms = 0;

		int[][] resultBestBlocks = null;
		double tempBestEntropy = Double.MAX_VALUE;
		int[][] tempNewBlocks = null;
		double tempEntropy;
		for (int i = 0; i < NUM_ALGORITHMS; i++) {
			// Is this algorithm available?
			if (!availableAlgorithms[i]) {
				continue;
			} // Of if

			try {
				SimpleTools.processTrackingOutput("Cluster in two using algorithm " + i + "... ");
				tempNewBlocks = clusterInTwo(paraBlock, i);
				SimpleTools.processTrackingOutput(" done.\r\n");
			} catch (UnableToClusterInKException ee) {
				System.out.println(ee);
				// Some algorithm may fail on some blocks.
				continue;
			} // Of try

			if (areBlocksImbalance(tempNewBlocks)) {
				// Do not consider unbalanced blocks.
				System.out.println("Algorithm " + i + " produces imbalance blocks with length "
						+ tempNewBlocks[0].length + " vs. " + tempNewBlocks[1].length + ".\r\n");
				continue;
			} // Of if

			tempEntropy = computeWeightedEntropy(tempNewBlocks);
			// tempEntropy = computeEntropy(tempNewBlocks);

			System.out.println("Algorithm " + i + " entropy: " + tempEntropy + " for " + paraBlock.length
					+ " instances splitted to " + tempNewBlocks[0].length + " + " + tempNewBlocks[1].length);
			if (tempBestEntropy > tempEntropy) {
				numBestAlgorithms = 0;
				bestAlgorithmArray[numBestAlgorithms] = i;
				numBestAlgorithms++;

				tempBestEntropy = tempEntropy;
				resultBestBlocks = tempNewBlocks;
			} else if (Math.abs(tempBestEntropy - tempEntropy) < 1e-6) {
				bestAlgorithmArray[numBestAlgorithms] = i;
				numBestAlgorithms++;
			} // Of if
		} // Of for i
		System.out.println("numBestAlgorithms = " + numBestAlgorithms + " in computeBestSubBlocks()");
		System.out.println("The best algorithms are " + Arrays.toString(bestAlgorithmArray));

		if (resultBestBlocks == null) {
			throw new UnableToClusterInKException(
					"Error occurred in computeBestSubBlocks(). No algortihm can handle this block: "
							+ Arrays.toString(paraBlock));
		} // Of if

		return resultBestBlocks;
	}// Of computeBestSubBlocks

	/**
	 ************************* 
	 * Learn. The most important process. This method takes advantage of a queue
	 * for unfinished blocks. The provided labels are used up.
	 * 
	 * @return The result information.
	 ************************* 
	 */
	public String learn() {
		SimpleTools.processTrackingOutput("SpecifiedLabelsAlgorithmSelection.learn(), numRemainingLabels = "
				+ numRemainingQueries + ", numRepresentativeLabels = " + numRepresentativeLabels + "\r\n");
		SimpleTools.processTrackingOutput("\r\n smallBlockThreshold = " + smallBlockThreshold); 

		// Step 1. Initialize
		Arrays.fill(instanceStatusArray, UNHANDLED);
		finalNumBlocks = 0;
		numSmallBlocks = 0;
		int tempNumPureBlocks = 0;
		int tempPureBlocksSizeSum = 0;
		Arrays.fill(algorithmWinArray, 0);

		// Step 2. Select representative instances to label
		int tempStrategy = instanceSelectionStrategy;
		instanceSelectionStrategy = DP_REPRESENTATIVE;
		try {
			SimpleTools.processTrackingOutput(
					"Trying to label " + numRepresentativeLabels + " representative instances ... ");
			selectCriticalAndLabel(wholeBlock, numRepresentativeLabels);
		} catch (LabelUsedUpException ee1) {
			SimpleTools.consoleOutput(ee1.toString());
		} catch (Exception ee2) {
			System.out.println("Internal error: " + ee2);
			ee2.printStackTrace();
			System.exit(0);
		} // Of try

		// Now restore.
		instanceSelectionStrategy = tempStrategy;

		// Step 3. Active learning with a queue
		BlockQueue tempQueue = new BlockQueue();
		tempQueue.enqueue(wholeBlock);
		SimpleTools.processTrackingOutput("Enqueue " + wholeBlock.length + " (" + wholeBlock[0] + ", ...); ");
		int[] tempBlock = null;
		//int tempNumRequiredLabels;
		boolean tempLabelUsedUp = false;

		while (!tempQueue.isEmpty()) {
			// Step 3.1 Take out a block to process
			tempBlock = tempQueue.dequeue();
			SimpleTools.processTrackingOutput("\r\nDequeue " + tempBlock.length + " (" + tempBlock[0] + ", ...); ");

			// Step 3.2 Select critical instances to label
			//tempNumRequiredLabels = getBlockPureThreshold(tempBlock.length);
			try {
				selectCriticalAndLabel(tempBlock);
			} catch (LabelUsedUpException ee1) {
				SimpleTools.consoleOutput(ee1.toString());
				finalNumBlocks ++;
				break;
			} catch (Exception ee2) {
				System.out.println("Internal error: " + ee2);
				ee2.printStackTrace();
				System.exit(0);
			} // Of try

			// Step 3.3 Classify if pure
			boolean tempPure = classifyIfPure(tempBlock);
			if (tempPure) {
				tempNumPureBlocks++;
				tempPureBlocksSizeSum += tempBlock.length;
				SimpleTools.processTrackingOutput(
						"\r\nPure block " + tempBlock.length + " (" + tempBlock[0] + ", ...) \r\n");

				finalNumBlocks++;
				SimpleTools.processTrackingOutput("Blocks " + finalNumBlocks + "(" + tempBlock.length + ")");
				continue;
			} // Of if

			// Step 3.4 Split in two using the best algorithm
			// Maybe more than one best algorithms
			int[][] tempBestBlocks = null;
			try {
				tempBestBlocks = computeBestSubBlocks(tempBlock);
			} catch (UnableToClusterInKException ee) {
				System.out.println(
						"No algorithm can handle this block: " + tempBlock.length + " (" + tempBlock[0] + "...)");
				//oneNnUnhandled(tempBlock);
				finalNumBlocks ++;
				System.out.println("finalNumBlocks = " + finalNumBlocks);
				SimpleTools.processTrackingOutput("Blocks " + finalNumBlocks + "(" + tempBlock.length + ")");
				continue;
			} // Of try

			int tempFirstRoundBestAlgorithm = bestAlgorithmArray[0];

			if (retrospect) {
				SimpleTools.processTrackingOutput("retrospect");
				// Step 3.5 Retrospect. Important code, different from the
				// superclass.
				for (int i = 0; i < 2; i++) {
					//tempNumRequiredLabels = getBlockPureThreshold(tempBestBlocks[i].length);
					try {
						selectCriticalAndLabel(tempBestBlocks[i]);
					} catch (LabelUsedUpException ee1) {
						SimpleTools.consoleOutput(ee1.toString());
						tempLabelUsedUp = true;
						break;
					} catch (Exception ee2) {
						System.out.println("Internal error: " + ee2);
						ee2.printStackTrace();
						System.exit(0);
					} // Of try
				} // Of for i

				if (tempLabelUsedUp) {
					break;
				} // Of if

				// Step 3.6 This time let's really do it. Important code,
				int[][] tempSecondRoundBestBlocks = null;
				try {
					tempSecondRoundBestBlocks = computeBestSubBlocks(tempBlock);
					//Replace it if success.
					tempBestBlocks = tempSecondRoundBestBlocks;
				} catch (UnableToClusterInKException ee) {
					System.out.println("Internal error: UnableToClusterInK in the retrospect stage. Use that of the first round.");
				} // Of try
				
				int tempSecondRoundBestAlgorithm = bestAlgorithmArray[0];
				if (tempFirstRoundBestAlgorithm != tempSecondRoundBestAlgorithm) {
					System.out.println("We have changed our mind from algorithm " + tempFirstRoundBestAlgorithm + " to "
							+ tempSecondRoundBestAlgorithm);
				} // Of if
			} // Of if retrospect
			
			for (int i = 0; i < numBestAlgorithms; i++) {
				algorithmWinArray[bestAlgorithmArray[i]]++;
			} // Of for i

			SimpleTools.processTrackingOutput("Splitting a block with " + tempBlock.length
					+ " instances. The best algorithm is: " + bestAlgorithmArray[0]);
			SimpleTools.consoleOutput(Arrays.deepToString(tempBestBlocks));

			// Step 3.5 Learn these two blocks.
			for (int i = 0; i < 2; i++) {
				if (tempBestBlocks[i].length > smallBlockThreshold) {
					SimpleTools.processTrackingOutput(
							"Enqueue " + tempBestBlocks[i].length + " (" + tempBestBlocks[i][0] + ", ...); ");
					tempQueue.enqueue(tempBestBlocks[i]);
				} else {
					smallBlocks[numSmallBlocks] = tempBestBlocks[i];
					numSmallBlocks++;
					finalNumBlocks ++;
					System.out.println("one more small, finalNumBlocks = " + finalNumBlocks);

					SimpleTools.processTrackingOutput("Blocks " + finalNumBlocks + "(" + tempBlock.length + ")");
				} // Of if
			} // Of for i

			/*
			// Step 3.6 Any more labels?
			if (numRemainingQueries <= 0) {
				break;
			} // Of if
			*/
		} // Of while
		
		System.out.println("Still in the queue blocks: " + tempQueue.getLength());
		finalNumBlocks += tempQueue.getLength();
		
		// Step 6. Deal with unhandled instances.
		SimpleTools.processTrackingOutput("\r\nNow handle other instances using kNN.");
		//Query enough instances.
		boolean tempHasUnhandled = false;
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] == UNHANDLED) {
				tempHasUnhandled = true;
				break;
			}//Of if
		}//Of for i
		
		if (tempHasUnhandled) {
			//Step 6.1 Form a block of all unhandled instances.
			//Scan once to obtain the size.
			int tempSize = 0;
			for (int i = 0; i < numInstances; i++) {
				if (instanceStatusArray[i] == UNHANDLED) {
					tempSize ++;
				}//Of if
			}//Of for i

			//Scan another time to form it.
			int[] tempAllUnhandled = new int[tempSize];
			tempSize = 0;
			for (int i = 0; i < numInstances; i++) {
				if (instanceStatusArray[i] == UNHANDLED) {
					tempAllUnhandled[tempSize] = i;
					tempSize ++;
				}//Of if
			}//Of for i
			
			SimpleTools.processTrackingOutput("\r\n numRemainingQueries = " + numRemainingQueries);
			SimpleTools.processTrackingOutput("\r\n num unhandled instances = " + tempSize);
			//Step 6.2 Query the remaining number of labels.
			try {
				selectCriticalAndLabel(tempAllUnhandled, numRemainingQueries);
				//selectCriticalAndLabel(tempAllUnhandled, numRepresentativeLabels);
				//selectCriticalDensityPeaks(tempAllUnhandled, numRemainingQueries);
			} catch (Exception ee) {
				System.out.println("Error occurred in SpecifiedLabelsAlgorithmSelection.learn() while trying to invoke selectCriticalDensityPeaks(): " + ee);
			}//Of try
			
			//Step 6.3 Classify all unhandled instances.
			SimpleTools.processTrackingOutput("\r\nkNN classification, k = " + kValue);
			knnUnhandled();
			//oneNnUnhandled(wholeBlock);
			//oneNnUnhandled(tempAllUnhandled);
		}//Of if
		
		SimpleTools.processTrackingOutput("\r\nQuery sequence: " + Arrays.toString(getQuerySequence()) + "\r\n");

		String resultMessage = "";
		resultMessage += getNumQueries(); // numQueries
		resultMessage += ", accuracy = " + computeAccuracy(); // accuracy
		resultMessage += ", algorithm wins = [";
		for (int i = 0; i < algorithmWinArray.length; i++) {
			resultMessage += "" + algorithmWinArray[i] + ", ";
		} // Of for i
		resultMessage += "], final number of blocks = " + finalNumBlocks;
		resultMessage += ", number of pure blocks = " + tempNumPureBlocks;
		resultMessage += ", pure blocks size sum = " + tempPureBlocksSizeSum;
		resultMessage += ", number of small blocks = " + numSmallBlocks;
		resultMessage += ", misclassified = " + getNumMisclassified();
		resultMessage += ", misclassified in pure blocks = " + getNumMisclassified(PURE_BLOCK_PREDICTED);
		resultMessage += ", misclassified by kNN = " + getNumMisclassified(KNN_PREDICTED);
		resultMessage += ", misclassified by default label = " + getNumMisclassified(DEFAULT_LABELED);
		resultMessage += ", numDefaultLabeled = " + getNumInstancesByStatus(DEFAULT_LABELED);
		resultMessage += ", unhandled = " + Arrays.toString(getInstancesByStatus(UNHANDLED));

		return resultMessage;
	}// Of learn

	/**
	 ************************* 
	 * Learn. The most important process. This method takes advantage of a queue
	 * for unfinished blocks.
	 * 
	 * @return The result information.
	 ************************* 
	 */
	public String learnDeprecated() {
		SimpleTools.processTrackingOutput("SpecifiedLabelsAlgorithmSelection.learn(), numRemainingLabels = "
				+ numRemainingQueries + ", numRepresentativeLabels = " + numRepresentativeLabels + "\r\n");

		// Step 1. Initialize
		Arrays.fill(instanceStatusArray, UNHANDLED);
		numSmallBlocks = 0;
		int tempNumPureBlocks = 0;
		int tempPureBlocksSizeSum = 0;
		Arrays.fill(algorithmWinArray, 0);

		// Step 2. Select representative instances to label
		int tempStrategy = instanceSelectionStrategy;
		instanceSelectionStrategy = DP_REPRESENTATIVE;
		try {
			SimpleTools.processTrackingOutput(
					"Trying to label " + numRepresentativeLabels + " representative instances ... ");
			selectCriticalAndLabel(wholeBlock, numRepresentativeLabels);
		} catch (LabelUsedUpException ee1) {
			SimpleTools.consoleOutput(ee1.toString());
		} catch (Exception ee2) {
			System.out.println("Internal error: " + ee2);
			ee2.printStackTrace();
			System.exit(0);
		} // Of try

		// Now restore.
		instanceSelectionStrategy = tempStrategy;

		// Step 3. Select representative/far instances to label
		BlockQueue tempQueue = new BlockQueue();
		tempQueue.enqueue(wholeBlock);
		SimpleTools.processTrackingOutput("Enqueue " + wholeBlock.length + " (" + wholeBlock[0] + ", ...); ");
		int[] tempBlock = null;
		int tempNumRequiredLabels;
		boolean tempCurrentBlockUnfinished = false;
		boolean tempLabelUsedUp = false;

		while (!tempQueue.isEmpty()) {
			// Step 3.1 Take out a block to process
			tempBlock = tempQueue.dequeue();
			SimpleTools.processTrackingOutput("\r\nDequeue " + tempBlock.length + " (" + tempBlock[0] + ", ...); ");

			// Step 3.2 Select critical instances to label
			tempNumRequiredLabels = getBlockPureThreshold(tempBlock.length);
			try {
				selectCriticalAndLabel(tempBlock, tempNumRequiredLabels);
			} catch (LabelUsedUpException ee1) {
				SimpleTools.consoleOutput(ee1.toString());
				tempCurrentBlockUnfinished = true;
				break;
			} catch (Exception ee2) {
				System.out.println("Internal error: " + ee2);
				ee2.printStackTrace();
				System.exit(0);
			} // Of try

			// Step 3.3 Classify if pure
			boolean tempPure = classifyIfPure(tempBlock);
			if (tempPure) {
				tempNumPureBlocks++;
				tempPureBlocksSizeSum += tempBlock.length;
				SimpleTools.processTrackingOutput(
						"\r\nPure block " + tempBlock.length + " (" + tempBlock[0] + ", ...) \r\n");
				continue;
			} // Of if

			// Step 3.4 Split in two using the best algorithm
			// Maybe more than one best algorithms
			int[][] tempBestBlocks = null;
			try {
				tempBestBlocks = computeBestSubBlocks(tempBlock);
			} catch (UnableToClusterInKException ee) {
				System.out.println(
						"No algorithm can handle this block: " + tempBlock.length + " (" + tempBlock[0] + "...)");
				System.out.println(" Use 1NN!");
				oneNnUnhandled(tempBlock);
				continue;
			} // Of try

			int tempFirstRoundBestAlgorithm = bestAlgorithmArray[0];
			// System.out.println("numBestAlgorithms =" + numBestAlgorithms);

			// if (numBestAlgorithms == 0) {
			// System.exit(0);
			// } // Of if

			if (retrospect) {
				SimpleTools.processTrackingOutput("retrospect");
				// Step 3.5 Retrospect. Important code, different from the
				// superclass.
				for (int i = 0; i < 2; i++) {
					tempNumRequiredLabels = getBlockPureThreshold(tempBestBlocks[i].length);
					try {
						selectCriticalAndLabel(tempBestBlocks[i], tempNumRequiredLabels);
					} catch (LabelUsedUpException ee1) {
						SimpleTools.consoleOutput(ee1.toString());
						tempCurrentBlockUnfinished = true;
						tempLabelUsedUp = true;
						break;
					} catch (Exception ee2) {
						System.out.println("Internal error: " + ee2);
						ee2.printStackTrace();
						System.exit(0);
					} // Of try
				} // Of for i

				if (tempLabelUsedUp) {
					break;
				} // Of if

				// Step 3.6 This time let's really do it. Important code,
				// different from the superclass.
				try {
					tempBestBlocks = computeBestSubBlocks(tempBlock);
				} catch (UnableToClusterInKException ee) {
					System.out.println("Internal error: UnableToClusterInK in the retrospect stage.");
					System.exit(0);
				} // Of try
				int tempSecondRoundBestAlgorithm = bestAlgorithmArray[0];
				if (tempFirstRoundBestAlgorithm != tempSecondRoundBestAlgorithm) {
					System.out.println("We have changed our mind from algorithm " + tempFirstRoundBestAlgorithm + " to "
							+ tempSecondRoundBestAlgorithm);
				} // Of if

			} // Of if retrospect
			for (int i = 0; i < numBestAlgorithms; i++) {
				algorithmWinArray[bestAlgorithmArray[i]]++;
			} // Of for i

			SimpleTools.processTrackingOutput("Splitting a block with " + tempBlock.length
					+ " instances. The best algorithm is: " + bestAlgorithmArray[0]);
			SimpleTools.consoleOutput(Arrays.deepToString(tempBestBlocks));

			// Step 3.5 Learn these two blocks.
			for (int i = 0; i < 2; i++) {
				if (tempBestBlocks[i].length > smallBlockThreshold) {
					SimpleTools.processTrackingOutput(
							"Enqueue " + tempBestBlocks[i].length + " (" + tempBestBlocks[i][0] + ", ...); ");
					tempQueue.enqueue(tempBestBlocks[i]);
				} else {
					smallBlocks[numSmallBlocks] = tempBestBlocks[i];
					numSmallBlocks++;
				} // Of if
			} // Of for i

			// Step 3.6 Any more labels?
			if (numRemainingQueries <= 0) {
				break;
			} // Of if
		} // Of while

		if (tempCurrentBlockUnfinished) {
			SimpleTools.processTrackingOutput(
					"Unfinished block " + tempBlock.length + " (" + tempBlock[0] + ", ...) classfied by 1NN.\r\n");
			oneNnUnhandled(tempBlock);
		} // Of if

		// Step 4. Classify other instances even if the block is impure
		SimpleTools.processTrackingOutput("Now handle remaining blocks in the queue using 1NN.\r\n");
		while (!tempQueue.isEmpty()) {
			// Step 4.1 Take out a block to process
			tempBlock = tempQueue.dequeue();

			// Step 4.2 1NN prediction
			SimpleTools.processTrackingOutput(
					"Block " + tempBlock.length + " (" + tempBlock[0] + ", ...) classfied by 1NN.\r\n");
			oneNnUnhandled(tempBlock);
		} // Of while

		// Step 5. Handle small blocks.
		SimpleTools.processTrackingOutput("Now handle small blocks.\r\n");
		for (int i = 0; i < numSmallBlocks; i++) {
			SimpleTools.processTrackingOutput("" + smallBlocks[i].length + ", ");
			oneNnUnhandled(smallBlocks[i]);
		} // Of for i
		
		SimpleTools.processTrackingOutput("\r\nQuery sequence: " + Arrays.toString(getQuerySequence()) + "\r\n");

		String resultMessage = "";
		resultMessage += getNumQueries(); // numQueries
		resultMessage += ", accuracy = " + computeAccuracy(); // accuracy
		resultMessage += ", algorithm wins = [";
		for (int i = 0; i < algorithmWinArray.length; i++) {
			resultMessage += "" + algorithmWinArray[i] + ", ";
		} // Of for i
		resultMessage += "], final number of blocks = " + finalNumBlocks;
		resultMessage += ", number of pure blocks = " + tempNumPureBlocks;
		resultMessage += ", pure blocks size sum = " + tempPureBlocksSizeSum;
		resultMessage += ", number of small blocks = " + numSmallBlocks;
		resultMessage += ", misclassified = " + getNumMisclassified();
		resultMessage += ", misclassified in pure blocks = " + getNumMisclassified(PURE_BLOCK_PREDICTED);
		resultMessage += ", misclassified by kNN = " + getNumMisclassified(KNN_PREDICTED);
		resultMessage += ", misclassified by default label = " + getNumMisclassified(DEFAULT_LABELED);
		resultMessage += ", numDefaultLabeled = " + getNumInstancesByStatus(DEFAULT_LABELED);
		resultMessage += ", unhandled = " + Arrays.toString(getInstancesByStatus(UNHANDLED));

		return resultMessage;
	}// Of learnDeprecated	
}// Of class SpecifiedLabelsAlgorithmSelection
