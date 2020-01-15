package algorithm;

import java.util.Arrays;

import clustering.Clustering;
import clustering.SmaleDBScan;
import clustering.DensityPeaks;
import clustering.FCM;
import clustering.SmaleHierarchical;
import clustering.KMeans;
import clustering.RandomWalks;
import clustering.WekaDBScan;
import clustering.WekaHierarchical;
import common.BlockQueue;
import common.DistanceMeasure;
import common.SimpleTools;
import exception.*;

/**
 * Clustering-based active learning. The clustering algorithm should be
 * specified.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Almost finished, further revision is possible.<br>
 * Written time: June 20, 2019. <br>
 * Last modify time: July 21, 2019.
 */
public class ClusteringBasedActiveLearning extends ActiveLearning {
	/**
	 * The whole block.
	 */
	int[] wholeBlock;

	/**
	 * How many blocks has been obtained.
	 */
	public int finalNumBlocks;

	/**
	 * The threshold for small blocks. Small block should be handled
	 * differently.
	 */
	int smallBlockThreshold;

	/**
	 * The minimal threshold for tiny blocks. These blocks should not be
	 * generated.
	 */
	public static final int TINY_BLOCK_THRESHOLD = 2;

	/**
	 * The threshold for imbalance blocks. These blocks should not be generated.
	 */
	public static final int IMBALANCE_THRESHOLD = 100;

	/**
	 * The number of small blocks.
	 */
	int numSmallBlocks;

	/**
	 * Small blocks. They are handled at the end of the learning process.
	 */
	int[][] smallBlocks;

	/**
	 * The density peaks algorithm with Gaussian kernel.
	 */
	public static final int DP_GAUSSIAN_CLUSTERING = 0;

	/**
	 * The kMeans algorithm.
	 */
	public static final int KMEANS_CLUSTERING = 1;

	/**
	 * The hierarchical algorithm.
	 */
	public static final int HIERARCHICAL_CLUSTERING = 2;

	/**
	 * The DBSCan algorithm.
	 */
	public static final int DBSCAN_CLUSTERING = 3;

	/**
	 * The Fuzzy C-means algorithm.
	 */
	public static final int FCM_CLUSTERING = 4;

	/**
	 * The random walk algorithm.
	 */
	public static final int RANDOM_WALK_CLUSTERING = 5;

	/**
	 * The density peaks algorithm with cutoff kernel.
	 */
	public static final int DP_CUTOFF_CLUSTERING = 6;

	/**
	 * The number of algorithms.
	 */
	public static final int NUM_ALGORITHMS = 7;

	/**
	 * The current algorithm for clustering.
	 */
	public Clustering currentClusteringAlgorithm;

	/**
	 * The current algorithm for clustering (index).
	 */
	public int currentClusteringAlgorithmIndex;

	/**
	 * KMeans clustering algorithm.
	 */
	KMeans kMeans;

	/**
	 * Density peaks clustering algorithm with cutoff kernel.
	 */
	DensityPeaks densityPeaksCutoff;

	/**
	 * Density peaks clustering algorithm with Gaussian kernel.
	 */
	DensityPeaks densityPeaksGaussian;

	/**
	 * Hierarchical clustering algorithm.
	 */
	SmaleHierarchical hierarchical;

	/**
	 * DBScan clustering algorithm.
	 */
	SmaleDBScan dbScan;

	/**
	 * FCM clustering algorithm.
	 */
	FCM fcm;

	/**
	 * Random walk clustering algorithm.
	 */
	RandomWalks randomWalks;

	/**
	 * Query amount strategy.
	 */
	int queryAmountStrategy;

	/**
	 * Enough queries at a time.
	 */
	public static final int ENOUGH_QUERIES = 0;

	/**
	 * Enough queries or encounter impure label.
	 */
	public static final int IMPURE_QUERIES = 1;

	/**
	 * Instance selection strategy.
	 */
	int instanceSelectionStrategy;

	/**
	 * Density * distance. See the density peaks algorithm.
	 */
	public static final int DP_REPRESENTATIVE = 0;

	/**
	 * Maximal total distance from labeled ones.
	 */
	public static final int MAX_TOTAL_DISTANCE = 1;

	/**
	 * Adaptive ratio for radius setting.
	 */
	double adaptiveRatio;

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
	 * @param paraDisorderData
	 *            Disorder data or not.
	 * @param paraAdaptiveRatio
	 *            The adaptive ratio for radius computation.
	 * @param paraSmallBlockThreshold
	 *            Small block threshold.
	 * @param paraInstanceSelectionStrategy
	 *            The instance selection strategy.
	 * @param paraQueryAmountStrategy
	 *            The query amount strategy.
	 ********************
	 */
	public ClusteringBasedActiveLearning(String paraFilename, int paraDistanceMeasure, boolean paraNormalizeData,
			boolean paraDisorderData, double paraAdaptiveRatio, int paraSmallBlockThreshold,
			int paraInstanceSelectionStrategy, int paraQueryAmountStrategy) {
		super(paraFilename, paraDistanceMeasure, paraNormalizeData, paraDisorderData);

		setInstanceSelectionStrategy(paraInstanceSelectionStrategy);
		setQueryAmountStrategy(paraQueryAmountStrategy);
		adaptiveRatio = paraAdaptiveRatio;

		// Should be initialized in other places.
		wholeBlock = new int[numInstances];
		for (int i = 0; i < numInstances; i++) {
			wholeBlock[i] = i;
		} // Of for i

		finalNumBlocks = 0;

		smallBlockThreshold = (int) (numInstances * 0.0002);
		if (smallBlockThreshold < paraSmallBlockThreshold) {
			smallBlockThreshold = paraSmallBlockThreshold;
		} // Of if

		numRemainingQueries = 1000;

		numSmallBlocks = 0;
		smallBlocks = new int[numInstances][];
		// numDefaultLabeled = 0;

		kMeans = null;
		densityPeaksCutoff = null;
		// new DensityPeaks(data, distanceMeasure, 0, adaptiveRatio,
		// DensityPeaks.CUTOFF_KERNEL);
		densityPeaksGaussian = null;
		// new DensityPeaks(data, distanceMeasure, 0, adaptiveRatio,
		// DensityPeaks.GAUSSIAN_KERNEL);
		hierarchical = null;
		dbScan = null;
		fcm = null;
		randomWalks = null;
	}// Of the first constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraDistanceMeasure
	 *            The given distance measure in integer.
	 * @param paraAdaptiveRatio
	 *            The adaptive ratio for the neighborhood radius.
	 * @param paraSmallBlockThreshold
	 *            Small block threshold.
	 * @param paraInstanceSelectionStrategy
	 *            The instance selection strategy, representative-based,
	 *            distance-based.
	 * @param paraQueryAmountStrategy
	 *            The strategy for how labels are queried, ENOUGH_QUERIES or
	 *            IMPURE_QUERIES.
	 ********************
	 */
	public ClusteringBasedActiveLearning(String paraFilename, int paraDistanceMeasure, double paraAdaptiveRatio,
			int paraSmallBlockThreshold, int paraInstanceSelectionStrategy, int paraQueryAmountStrategy) {
		this(paraFilename, paraDistanceMeasure, true, false, paraAdaptiveRatio, paraSmallBlockThreshold,
				paraInstanceSelectionStrategy, paraQueryAmountStrategy);
	}// Of the second constructor

	/**
	 ************************* 
	 * Set the current algorithm.
	 * 
	 * @param paraAlgorithm
	 *            The given algorithm.
	 ************************* 
	 */
	public void setClusteringAlgorithm(int paraAlgorithm) {
		currentClusteringAlgorithmIndex = paraAlgorithm;
		currentClusteringAlgorithm = null;

		switch (paraAlgorithm) {
		case KMEANS_CLUSTERING:
			if (kMeans == null) {
				kMeans = new KMeans(data, distanceMeasure);
			} // Of if
			currentClusteringAlgorithm = kMeans;
			break;
		case FCM_CLUSTERING:
			if (fcm == null) {
				fcm = new FCM(data, distanceMeasure);
			} // Of if
			currentClusteringAlgorithm = fcm;
			break;
		case DBSCAN_CLUSTERING:
			if (dbScan == null) {
				dbScan = new SmaleDBScan(data, distanceMeasure, adaptiveRatio, DensityPeaks.CUTOFF_KERNEL);
			} // Of if
			currentClusteringAlgorithm = dbScan;
			break;
		case DP_CUTOFF_CLUSTERING:
			if (densityPeaksCutoff == null) {
				densityPeaksCutoff = new DensityPeaks(data, distanceMeasure, 0, adaptiveRatio,
						DensityPeaks.CUTOFF_KERNEL);
			} // Of if
			currentClusteringAlgorithm = densityPeaksCutoff;
			break;
		case DP_GAUSSIAN_CLUSTERING:
			if (densityPeaksGaussian == null) {
				densityPeaksGaussian = new DensityPeaks(data, distanceMeasure, 0, adaptiveRatio,
						DensityPeaks.GAUSSIAN_KERNEL);
			} // Of if
			currentClusteringAlgorithm = densityPeaksGaussian;
			break;
		case HIERARCHICAL_CLUSTERING:
			if (hierarchical == null) {
				hierarchical = new SmaleHierarchical(data, distanceMeasure);
			} // Of if
			currentClusteringAlgorithm = hierarchical;
			break;
		case RANDOM_WALK_CLUSTERING:
			if (randomWalks == null) {
				randomWalks = new RandomWalks(data, distanceMeasure);
			} // Of if
			currentClusteringAlgorithm = randomWalks;
			break;
		default:
			System.out.println("Unsupported algorithm: " + paraAlgorithm);
			System.exit(0);
		}// Of switch
	}// Of setClusteringAlgorithm

	/**
	 ************************* 
	 * Set the instance selection strategy.
	 * 
	 * @param paraInstanceSelectionStrategy
	 *            The given strategy.
	 ************************* 
	 */
	public void setInstanceSelectionStrategy(int paraInstanceSelectionStrategy) {
		instanceSelectionStrategy = paraInstanceSelectionStrategy;
	}// Of setInstanceSelectionStrategy

	/**
	 ************************* 
	 * Initialize densityPeaksGaussian for instance selection. It is more often
	 * employed for independent running based on one clustering algorithm such
	 * as kMeans. Please refer to the main method of this class.
	 ************************* 
	 */
	public void initializeDensityPeaksGaussian() {
		densityPeaksGaussian = new DensityPeaks(data, distanceMeasure, 0, adaptiveRatio, DensityPeaks.GAUSSIAN_KERNEL);
	}// Of initializeDensityPeaksGaussian

	/**
	 ************************* 
	 * Set the query amount strategy.
	 * 
	 * @param paraQueryAmountStrategy
	 *            The given strategy.
	 ************************* 
	 */
	public void setQueryAmountStrategy(int paraQueryAmountStrategy) {
		queryAmountStrategy = paraQueryAmountStrategy;
	}// Of setQueryAmountStrategy

	/**
	 ************************* 
	 * Select critical instances and label. The number of required labels is
	 * determined by the block size.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 * @see #getBlockPureThreshold(int);
	 ************************* 
	 */
	public void selectCriticalAndLabel(int[] paraBlock) throws LabelUsedUpException, DuplicateQueryException {
		int tempNumRequiredLabels = getBlockPureThreshold(paraBlock.length);
		selectCriticalAndLabel(paraBlock, tempNumRequiredLabels);
	}// Of selectCriticalAndLabel

	/**
	 ************************* 
	 * Select critical instances and label.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraTotalLabels
	 *            The number of labels (including already obtained ones).
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 */
	public void selectCriticalAndLabel(int[] paraBlock, int paraTotalLabels)
			throws LabelUsedUpException, DuplicateQueryException {
		// Leave small blocks along
		if (paraBlock.length <= smallBlockThreshold) {
			return;
		} // Of if
		
		//No more than the block length
		if (paraTotalLabels > paraBlock.length) {
			paraTotalLabels = paraBlock.length;
		}//Of if

		// Step 1. Check number of existing labels
		int tempNumExistingLabels = 0;
		boolean tempIsFirst = true;
		int tempFirstLabel = -1;
		int tempCurrentLabel;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				if (queryAmountStrategy == IMPURE_QUERIES) {
					if (tempIsFirst) {
						tempFirstLabel = predicts[paraBlock[i]];
						tempIsFirst = false;
					} else {
						tempCurrentLabel = predicts[paraBlock[i]];
						if (tempCurrentLabel != tempFirstLabel) {
							// Impure, do not select any instance
							return;
						} // Of if
					} // Of if
				} // Of if
				tempNumExistingLabels++;
			} // Of if
		} // Of for i

		SimpleTools.variableTrackingOutput("tempNumExistingLabels = " + tempNumExistingLabels);

		int tempRequiredLabels = paraTotalLabels - tempNumExistingLabels;
		SimpleTools.variableTrackingOutput("paraBlock length = " + paraBlock.length + ", paraTotalLabels = "
				+ paraTotalLabels + ", tempNumExistingLabels = " + tempNumExistingLabels + ", Require "
				+ tempRequiredLabels + " labels.");
		if (tempRequiredLabels <= 0) {
			return;
		} // Of if

		switch (instanceSelectionStrategy) {
		case DP_REPRESENTATIVE:
			selectCriticalDensityPeaks(paraBlock, tempRequiredLabels, tempIsFirst, tempFirstLabel);
			break;
		case MAX_TOTAL_DISTANCE:
			selectCriticalMaxTotalDistance(paraBlock, tempRequiredLabels, tempIsFirst, tempFirstLabel);
			break;
		default:
			System.out.println("Unsupported instance selection strategy: " + instanceSelectionStrategy);
			System.exit(0);
		}// Of switch
	}// Of selectCriticalAndLabel

	/**
	 ************************* 
	 * Select critical instances using the density peaks algorithm. Only support
	 * the ENOUGH_QUERY option.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraRequiredLabels
	 *            Required labels (not including already obtained ones).
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 */
	public void selectCriticalDensityPeaks(int[] paraBlock, int paraRequiredLabels)
			throws LabelUsedUpException, DuplicateQueryException {
		// Nothing to select.
		if (paraRequiredLabels < 1) {
			return;
		} // Of if

		SimpleTools.processTrackingOutput(
				"Plan to label " + paraRequiredLabels + " while the block has " + paraBlock.length + " instances.");

		// Queries in this method.
		int tempQueries = 0;

		if (paraRequiredLabels >= paraBlock.length) {
			for (int i = 0; i < paraBlock.length; i++) {
				try {
					query(paraBlock[i]);
				} catch (DuplicateQueryException ee) {
					// Ignore it.
				} // Of try

				tempQueries++;
			} // Of for i

			return;
		} // Of if

		// Enough critical instances to select.
		int[] tempCriticalInstances = densityPeaksGaussian.computeCriticalInstances(paraBlock, paraRequiredLabels);

		SimpleTools.variableTrackingOutput("paraRequiredLabels = " + paraRequiredLabels);

		for (int i = 0; i < tempCriticalInstances.length; i++) {
			if (instanceStatusArray[tempCriticalInstances[i]] == QUERIED) {
				continue;
			} // Of if

			try {
				query(tempCriticalInstances[i]);
			} catch (DuplicateQueryException ee) {
				// Ignore it.
			} // Of try

			tempQueries++;
			if (tempQueries >= paraRequiredLabels) {
				break;
			} // Of if
		} // Of for i
	}// Of selectCriticalDensityPeaks

	/**
	 ************************* 
	 * Select critical instances using the density peaks algorithm.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraRequiredLabels
	 *            Required labels (not including already obtained ones).
	 * @param paraIsFirst
	 *            Is the first labeled instance? Or has any instance be labeled
	 *            in this block?
	 * @param paraFirstLabel
	 *            The first label for deciding pure or not.
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 */
	public void selectCriticalDensityPeaks(int[] paraBlock, int paraRequiredLabels, boolean paraIsFirst,
			int paraFirstLabel) throws LabelUsedUpException, DuplicateQueryException {
		int tempPureThrehold = getBlockPureThreshold(paraBlock.length);
		// In the first round, we may query more than sqrt(n) labels.
		if (tempPureThrehold < paraRequiredLabels) {
			tempPureThrehold = paraRequiredLabels;
		} // Of if

		// Enough critical instances to select.
		int[] tempCriticalInstances = densityPeaksGaussian.computeCriticalInstances(paraBlock, tempPureThrehold);

		// Queries in this method.
		int tempQueries = 0;

		SimpleTools.variableTrackingOutput("paraRequiredLabels = " + paraRequiredLabels);
		SimpleTools.variableTrackingOutput("tempCriticalInstances.length = " + tempCriticalInstances.length);
		for (int i = 0; i < tempCriticalInstances.length; i++) {
			if (instanceStatusArray[tempCriticalInstances[i]] == QUERIED) {
				continue;
			} // Of if

			// SimpleTools.processTrackingOutput("Querying " +
			// tempCriticalInstances[i]
			// + "(DP), ");
			query(tempCriticalInstances[i]);
			tempQueries++;
			if (tempQueries >= paraRequiredLabels) {
				break;
			} // Of if
		} // Of for i
	}// Of selectCriticalDensityPeaks

	/**
	 ************************* 
	 * Select critical instances with the maximal distance from labeled ones.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraRequiredLabels
	 *            Required labels (not including already obtained ones).
	 * @param paraIsFirst
	 *            Is the first labeled instance? Or has any instance be labeled
	 *            in this block?
	 * @param paraFirstLabel
	 *            The first label for deciding pure or not.
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 */
	public void selectCriticalMaxTotalDistance(int[] paraBlock, int paraRequiredLabels, boolean paraIsFirst,
			int paraFirstLabel) throws LabelUsedUpException, DuplicateQueryException {
		// Nothing to select.
		if (paraRequiredLabels < 1) {
			return;
		} // Of if
		
		//No more than the size of the block.
		if (paraRequiredLabels > paraBlock.length) {
			paraRequiredLabels = paraBlock.length;
		}//Of if

		// Step 1. Obtain labeled instances.
		int tempNumLabeled = 0;
		int[] tempLabeled = new int[paraBlock.length];
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempLabeled[tempNumLabeled] = paraBlock[i];
				tempNumLabeled++;
			} // Of if
		} // Of for i

		SimpleTools.consoleOutput("selectCriticalMaxTotalDistance for a block with length " + paraBlock.length + " and "
				+ tempNumLabeled + " labeled instances.");

		// Step 2. Select and label
		int tempBestIndex;
		double tempMaxTotalDistance;
		for (int i = 0; i < paraRequiredLabels; i++) {
			tempBestIndex = -1;
			tempMaxTotalDistance = -1;
			for (int j = 0; j < paraBlock.length; j++) {
				if (instanceStatusArray[paraBlock[j]] == QUERIED) {
					continue;
				} // Of if
				double tempCurrentTotalDistance = 0;
				for (int k = 0; k < tempNumLabeled; k++) {
					tempCurrentTotalDistance += distanceMeasure.distance(paraBlock[j], paraBlock[k]);
				} // Of for k
				if (tempCurrentTotalDistance > tempMaxTotalDistance) {
					tempMaxTotalDistance = tempCurrentTotalDistance;
					tempBestIndex = j;
				} // Of if
			} // Of for j
				// Now label it.
			SimpleTools.consoleOutput("tempBestIndex = " + tempBestIndex);
			tempLabeled[tempNumLabeled] = paraBlock[tempBestIndex];
			tempNumLabeled++;

			// SimpleTools.processTrackingOutput("Querying " +
			// paraBlock[tempBestIndex] + "(MaxTotalDistance), ");

			// Now label it.
			query(paraBlock[tempBestIndex]);

			int tempCurrentLabel;
			if (queryAmountStrategy == IMPURE_QUERIES) {
				if (paraIsFirst) {
					paraFirstLabel = predicts[paraBlock[tempBestIndex]];
					paraIsFirst = false;
				} else {
					tempCurrentLabel = predicts[paraBlock[tempBestIndex]];
					if (tempCurrentLabel != paraFirstLabel) {
						// Impure, do not select any instance
						return;
					} // Of if
				} // Of if
			} // Of if
		} // Of for i
	}// Of selectCriticalMaxTotalDistance

	/**
	 ************************* 
	 * Cluster the given block in two using the given algorithm.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraAlgorithmIndex
	 *            The given algorithm.
	 * @return Two blocks
	 * @throws UnableToClusterInKException
	 *             If the clustered result contains only one block (the other is
	 *             empty).
	 ************************* 
	 */
	public int[][] clusterInTwo(int[] paraBlock, int paraAlgorithmIndex) throws UnableToClusterInKException {
		setClusteringAlgorithm(paraAlgorithmIndex);
		return clusterInTwo(paraBlock);
	}// Of clusterInTwo

	/**
	 ************************* 
	 * Cluster the given block in two using density peaks.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return Two blocks
	 ************************* 
	 */
	public int[] getQueriedArray(int[] paraBlock) {
		int[] tempQueriedArray = new int[paraBlock.length];
		int tempSize = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempQueriedArray[tempSize] = i;
				tempSize++;
			} // Of if
		} // Of for i
		int[] resultArray = new int[tempSize];
		for (int i = 0; i < tempSize; i++) {
			resultArray[i] = tempQueriedArray[i];
		} // Of for i

		return resultArray;
	}// Of getQueriedArray

	/**
	 ************************* 
	 * Cluster the given block in two using density peaks.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return Two blocks
	 * @throws UnableToClusterInKException
	 *             If the clustered result contains only one block (the other is
	 *             empty).
	 ************************* 
	 */
	public int[][] clusterInTwo(int[] paraBlock) throws UnableToClusterInKException {
		int[] tempQueriedArray = getQueriedArray(paraBlock);
		int[][] resultBlocks = currentClusteringAlgorithm.clusterInTwo(paraBlock, tempQueriedArray);

		if (resultBlocks == null) {
			throw new UnableToClusterInKException(
					"Error occurred in ClusteringBasedActiveLearning.clusterInTwo(int[]):\r\n" + "Algorithm #"
							+ currentClusteringAlgorithmIndex + " cannot handle this block: "
							+ Arrays.toString(paraBlock));
		} // Of if

		System.out.println("Block lengths = " + resultBlocks[0].length + ", " + resultBlocks[1].length);

		if ((resultBlocks[0].length == 0) || (resultBlocks[1].length == 0)) {
			throw new UnableToClusterInKException(
					"Error occurred in ClusteringBasedActiveLearning.clusterInTwo(int[]):\r\n" + "Algorithm #"
							+ currentClusteringAlgorithmIndex + " obtains the whole block and an empty block: "
							+ Arrays.toString(paraBlock));
		} // Of if

		return resultBlocks;
	}// Of clusterInTwo

	/**
	 ************************* 
	 * Label small block directly.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 *             public void labelSmallBlock(int[] paraBlock) throws
	 *             LabelUsedUpException, DuplicateQueryException { for (int i =
	 *             0; i < paraBlock.length; i++) { if
	 *             (instanceStatusArray[paraBlock[i]] == QUERIED) { // if
	 *             (isQueried(paraBlock[i])) { continue; } // Of if
	 * 
	 *             query(paraBlock[i]); } // Of for i
	 * 
	 *             finalNumBlocks++; }// Of labelSmallBlock
	 */

	/**
	 ************************* 
	 * Classify this block if it has labeled instances and is pure.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return Pure or not.
	 ************************* 
	 */
	boolean classifyIfPure(int[] paraBlock) {
		// Step 1. Obtain the first label.
		int tempFirstLabel = -1;
		int tempFirstLabelIndex = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempFirstLabel = predicts[paraBlock[i]];
				tempFirstLabelIndex = i;
				break;
			} // Of if
		} // Of for i

		if (tempFirstLabel == UNHANDLED) {
			// No label has been handled/queried in this block.
			return false;
		} // Of if

		// Step 2. Check the purity.
		for (int i = tempFirstLabelIndex + 1; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				if (predicts[paraBlock[i]] != tempFirstLabel) {
					// It is impure.
					SimpleTools.variableTrackingOutput("" + paraBlock[i] + ": paraBlock[" + i + "] has label "
							+ predicts[paraBlock[i]] + " different from " + tempFirstLabel);
					return false;
				} // Of if
			} // Of if
		} // Of for i

		// Step 3. Classify.
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] != QUERIED) {
				predicts[paraBlock[i]] = tempFirstLabel;
				changeInstanceStatus(paraBlock[i], PURE_BLOCK_PREDICTED);
			} // Of if
		} // Of for i

		return true;
	}// Of classifyIfPure

	/**
	 ************************* 
	 * Classify this block by voting. It is not a good strategy. We may use 1NN
	 * instead. In case no label has been queried, the default label will be 0.
	 * 
	 * @param paraBlock
	 *            The given block.
	 ************************* 
	 */
	void classifyByVotings(int[] paraBlock) {
		int[] tempCountArray = new int[data.classAttribute().numValues()];

		// Step 1. Statistics
		for (int i = 0; i < paraBlock.length; i++) {
			if (predicts[paraBlock[i]] >= 0) {
				tempCountArray[predicts[paraBlock[i]]]++;
			} // Of if
		} // Of for i

		// Step 2. Find the majority
		int tempMajority = 0;
		int tempMaximal = tempCountArray[0];
		for (int i = 1; i < tempCountArray.length; i++) {
			if (tempMaximal < tempCountArray[i]) {
				tempMaximal = tempCountArray[i];
				tempMajority = i;
			} // Of if
		} // Of for i

		// Step 3. Classify others
		for (int i = 0; i < paraBlock.length; i++) {
			if (predicts[paraBlock[i]] < 0) {
				predicts[paraBlock[i]] = tempMajority;
			} // Of if
		} // Of for i
	}// Of classifyByVotings

	/**
	 ************************* 
	 * Get the number of blocks.
	 * 
	 * @return The the number of blocks.
	 ************************* 
	 */
	public int getNumBlocks() {
		return finalNumBlocks;
	}// Of getNumBlocks

	/**
	 ************************* 
	 * How many labels are enough to say that the block is pure? Now the
	 * strategy is sqrt(n). Attention: We may implement other strategies in the
	 * future.
	 * 
	 * @param paraBlockLength
	 *            The length of the block.
	 * @return The threshold.
	 ************************* 
	 */
	public int getBlockPureThreshold(int paraBlockLength) {
		int tempThreshold = (int) Math.sqrt(paraBlockLength);
		return tempThreshold;
	}// Of getBlockPureThreshold

	/**
	 ************************* 
	 * Classify unhandled instances using 1NN. The neighbor should be in the
	 * same block. In case no label has been queried, the default label will be
	 * 0.
	 * 
	 * @param paraBlock
	 *            The given block.
	 ************************* 
	 */
	public void oneNnUnhandled(int[] paraBlock) {
		// Step 1. How many instances have been queried?
		int tempNumQueried = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempNumQueried++;
			} // Of if
		} // Of for i

		// Step 2. Use default label if no queried labels.
		if (tempNumQueried == 0) {
			try {
				// Label one and classify others if there is still label to use.
				query(paraBlock[0]);
				int tempLabel = predicts[paraBlock[0]];
				for (int i = 1; i < paraBlock.length; i++) {
					predicts[paraBlock[i]] = tempLabel;
					changeInstanceStatus(paraBlock[i], KNN_PREDICTED);
				} // Of for i
			} catch (LabelUsedUpException ee1) {
				// No more query, so use default label.
				for (int i = 0; i < paraBlock.length; i++) {
					predicts[paraBlock[i]] = DEFAULT_LABEL;
					changeInstanceStatus(paraBlock[i], DEFAULT_LABELED);
				} // Of for i
			} catch (DuplicateQueryException ee2) {
				System.out.println("Internal error occurred in ClusteringBasedActiveLearning.oneNnUnhandled(int[])");
			} // Of try

			return;
		} // Of if

		// Step 3. Construct the queried array. This approach saves time when
		// the block is big and tempNumQueried is small.
		int[] tempQueriedArray = new int[tempNumQueried];
		int tempCounter = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] == QUERIED) {
				tempQueriedArray[tempCounter] = paraBlock[i];
				tempCounter++;
			} // Of if
		} // Of for i

		// Step 3. Find the nearest neighbor.
		double tempMinDistance;
		double tempDistance;
		int tempClosest;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStatusArray[paraBlock[i]] != UNHANDLED) {
				continue;
			} // Of if

			tempMinDistance = Double.MAX_VALUE;
			tempClosest = -1;

			for (int j = 0; j < tempNumQueried; j++) {
				tempDistance = distanceMeasure.distance(paraBlock[i], tempQueriedArray[j]);
				if (tempMinDistance > tempDistance) {
					tempMinDistance = tempDistance;
					tempClosest = tempQueriedArray[j];
				} // Of if
			} // Of for j

			predicts[paraBlock[i]] = predicts[tempClosest];
			changeInstanceStatus(paraBlock[i], KNN_PREDICTED);
		} // Of for i
	}// Of oneNnUnhandled

	/**
	 ************************* 
	 * Learn. The most important process. This method takes advantage of a queue
	 * for unfinished blocks.
	 * 
	 * @return The result information.
	 ************************* 
	 */
	public String learn() {
		// Step 1. Initialize. Half are representative instances. Test only. May
		// change later.
		Arrays.fill(instanceStatusArray, UNHANDLED);
		numSmallBlocks = 0;
		int tempNumRepresentativeLabels = numRemainingQueries / 2;
		int tempNumPureBlocks = 0;
		int tempPureBlocksSizeSum = 0;

		// Step 2. Select representative instances to label
		int tempStrategy = instanceSelectionStrategy;
		instanceSelectionStrategy = DP_REPRESENTATIVE;
		try {
			SimpleTools.processTrackingOutput(
					"Trying to label " + tempNumRepresentativeLabels + " representative instances ... ");
			selectCriticalAndLabel(wholeBlock, tempNumRepresentativeLabels);
			SimpleTools.processTrackingOutput("done. \r\n");
		} catch (LabelUsedUpException ee1) {
			System.out.println(ee1.toString());
		} catch (Exception ee2) {
			System.out.println("Internal error: " + ee2);
			ee2.printStackTrace();
			System.exit(0);
		} // Of try

		// Now restore.
		instanceSelectionStrategy = tempStrategy;

		// Step 3. Select edge/far instances to label
		BlockQueue tempQueue = new BlockQueue();
		tempQueue.enqueue(wholeBlock);
		SimpleTools.processTrackingOutput("Enqueue " + wholeBlock.length + " (" + wholeBlock[0] + ", ...); ");
		int[] tempBlock = null;
		int tempAvailableLabels;
		boolean tempCurrentBlockUnfinished = false;

		while (!tempQueue.isEmpty()) {
			// Step 3.1 Take out a block to process
			tempBlock = tempQueue.dequeue();
			SimpleTools.processTrackingOutput("Dequeue " + tempBlock.length + " (" + tempBlock[0] + ", ...); ");

			// Step 3.2 Select critical instances to label
			tempAvailableLabels = getBlockPureThreshold(tempBlock.length);
			try {
				selectCriticalAndLabel(tempBlock, tempAvailableLabels);
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

			// Step 3.4 Split in two using the current algorithm
			int[][] tempNewBlocks = null;
			try {
				tempNewBlocks = clusterInTwo(tempBlock);
			} catch (UnableToClusterInKException ee) {
				// Attention: Some algorithm may fail on some blocks. Use 1NN.
				oneNnUnhandled(tempBlock);
				continue;
			} // Of try

			// Step 3.5 Learn these two blocks.
			if (tempNewBlocks[0].length > smallBlockThreshold) {
				tempQueue.enqueue(tempNewBlocks[0]);
				SimpleTools.processTrackingOutput(
						"Enqueue " + tempNewBlocks[0].length + " (" + tempNewBlocks[0][0] + ", ...); ");
			} else {
				smallBlocks[numSmallBlocks] = tempNewBlocks[0];
				numSmallBlocks++;
			} // Of if

			if (tempNewBlocks[1].length > smallBlockThreshold) {
				SimpleTools.processTrackingOutput(
						"Enqueue " + tempNewBlocks[1].length + " (" + tempNewBlocks[1][0] + ", ...); ");
				tempQueue.enqueue(tempNewBlocks[1]);
			} else {
				smallBlocks[numSmallBlocks] = tempNewBlocks[1];
				numSmallBlocks++;
			} // Of if

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
			SimpleTools.processTrackingOutput("" + tempBlock.length + ", ");
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
		resultMessage += "Clustering algorithm index = " + currentClusteringAlgorithmIndex;
		resultMessage += ": " + currentClusteringAlgorithm;
		resultMessage += ", queries = " + getNumQueries(); // numQueries
		resultMessage += ", accuracy = " + computeAccuracy(); // accuracy
		resultMessage += ", final number of blocks = " + finalNumBlocks;
		resultMessage += ", number of pure blocks = " + tempNumPureBlocks;
		resultMessage += ", pure blocks size sum = " + tempPureBlocksSizeSum;
		resultMessage += ", number of small blocks = " + numSmallBlocks;
		resultMessage += ", misclassified = " + getNumMisclassified();
		resultMessage += ", misclassified in pure blocks = " + getNumMisclassified(PURE_BLOCK_PREDICTED);
		resultMessage += ", misclassified by 1NN = " + getNumMisclassified(KNN_PREDICTED);
		resultMessage += ", misclassified by default label = " + getNumMisclassified(DEFAULT_LABELED);
		resultMessage += ", numDefaultLabeled = " + getNumInstancesByStatus(DEFAULT_LABELED);
		resultMessage += ", unhandled = " + Arrays.toString(getInstancesByStatus(UNHANDLED));

		return resultMessage;
	}// Of learn

	/**
	 ************************* 
	 * 1NN classification using the same queries as the active learner. For test
	 * only!
	 * 
	 * @return The accuracy.
	 ************************* 
	 */
	public double sameQueriesOneNn() {
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] != QUERIED) {
				instanceStatusArray[i] = UNHANDLED;
			} // Of if
		} // Of for i
		oneNnUnhandled(wholeBlock);

		return computeAccuracy();
	}// Of sameQueriesOneNn

	/**
	 ************************* 
	 * Are the given blocks imbalance?
	 * 
	 * @param paraBlocks
	 *            The given blocks. Should be exactly two blocks.
	 * @return Imbalance or not.
	 ************************* 
	 */
	public static boolean areBlocksImbalance(int[][] paraBlocks) {
		// Step 1. Check tiny blocks.
		if ((paraBlocks[0].length <= TINY_BLOCK_THRESHOLD) || (paraBlocks[1].length <= TINY_BLOCK_THRESHOLD)) {
			return true;
		} // Of if

		// Step 2. Check imbalance blocks.
		double tempProportion;
		if (paraBlocks[0].length < paraBlocks[1].length) {
			tempProportion = (paraBlocks[1].length + 0.0) / paraBlocks[0].length;
		} else {
			tempProportion = (paraBlocks[0].length + 0.0) / paraBlocks[1].length;
		} // Of if

		if (tempProportion >= IMBALANCE_THRESHOLD) {
			return true;
		} // Of if

		return false;
	}// Of areBlocksImbalance

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
		System.out.println("Hello, clustering-based active learning. Only one base clustering technique is tested.");
		// String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/spiral.arff";
		String tempFilename = "src/data/mushroom.arff";
		// String tempFilename = "src/data/DLA.arff";
		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		ClusteringBasedActiveLearning tempLearner = new ClusteringBasedActiveLearning(tempFilename,
				DistanceMeasure.EUCLIDEAN, true, true, 0.03, 10, 0, 0);
		tempLearner.initializeDensityPeaksGaussian();
		// tempLearner.setClusteringAlgorithm(KMEANS_CLUSTERING);
		tempLearner.setClusteringAlgorithm(DP_GAUSSIAN_CLUSTERING);
		// tempLearner.setClusteringAlgorithm(DP_CUTOFF_CLUSTERING);
		// tempLearner.setClusteringAlgorithm(RANDOM_WALK_CLUSTERING);
		// tempLearner.setClusteringAlgorithm(FCM_CLUSTERING);

		tempLearner.setQueryFraction(0.06);
		System.out.println("Before learn()");
		String tempResults = tempLearner.learn();
		System.out.println(tempResults);

		System.out.println("after learn()");
		tempResults = "" + tempLearner.sameQueriesOneNn();
		System.out.println("What if we use the same queries for 1NN? Accuracy = " + tempResults);
	}// Of main
}// Of class ClusteringBasedActiveLearning
