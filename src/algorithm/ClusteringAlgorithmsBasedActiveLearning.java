package algorithm;

import clustering.SmaleDBScan;
import clustering.DensityPeaks;
import clustering.FCM;
import clustering.SmaleHierarchical;
import clustering.KMeans;
import clustering.RandomWalks;
import clustering.WekaDBScan;
import clustering.WekaHierarchical;
import common.*;

/**
 * Clustering algorithms based active learning. A number of clustering
 * algorithms will be employed.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Almost finished, further revision is possible.<br>
 * Written time: July 21, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public class ClusteringAlgorithmsBasedActiveLearning extends
		ClusteringBasedActiveLearning {

	/**
	 * Available algorithms.
	 */
	boolean[] availableAlgorithms;

	/**
	 * Statistics on algorithm usage.
	 */
	int[] algorithmWinArray;

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
	 *            Small block threshold.
	 * @param paraInstanceSelectionStrategy
	 *            The instance selection strategy.
	 * @param paraDisorderData
	 *            Disorder data or not.
	 * @param paraQueryAmountStrategy
	 *            The query amount strategy.
	 ********************
	 */
	public ClusteringAlgorithmsBasedActiveLearning(String paraFilename,
			int paraDistanceMeasure, boolean paraNormalizeData,
			boolean paraDisorderData, double paraAdaptiveRatio,
			int paraSmallBlockThreshold, int paraInstanceSelectionStrategy,
			int paraQueryAmountStrategy) {
		super(paraFilename, paraDistanceMeasure, paraNormalizeData,
				paraDisorderData, paraAdaptiveRatio, paraSmallBlockThreshold,
				paraInstanceSelectionStrategy, paraQueryAmountStrategy);

		// Prepare
		// setDc(0.5);
		availableAlgorithms = new boolean[NUM_ALGORITHMS];
		algorithmWinArray = new int[NUM_ALGORITHMS];
	}// Of the constructor

	/**
	 ********************
	 * Reset for repeated running.
	 ********************
	 */
	public void reset() {
		super.reset();

		SimpleTools
				.processTrackingOutput("initializePool for the next running\r\n");
		initializePool();
	}// Of reset

	/**
	 ********************
	 * Initialize the clustering algorithm pool.
	 ********************
	 */
	public void initializePool() {
		// DP-Gaussian must be initialized for critical instance selection.
		densityPeaksGaussian = new DensityPeaks(data, distanceMeasure, 0,
				adaptiveRatio, DensityPeaks.GAUSSIAN_KERNEL);
		SimpleTools
				.processTrackingOutput("densityPeaksGaussian initialized.\r\n");

		if (availableAlgorithms[KMEANS_CLUSTERING]) {
			kMeans = new KMeans(data, distanceMeasure);
			SimpleTools.processTrackingOutput("kMeans initialized.\r\n");
		}// Of if

		if (availableAlgorithms[HIERARCHICAL_CLUSTERING]) {
			hierarchical = new SmaleHierarchical(data, distanceMeasure);
			SimpleTools.processTrackingOutput("hierarchical initialized.\r\n");
		}// Of if

		if (availableAlgorithms[DBSCAN_CLUSTERING]) {
			dbScan = new SmaleDBScan(data, distanceMeasure, adaptiveRatio,
					DensityPeaks.CUTOFF_KERNEL);
			SimpleTools.processTrackingOutput("dbScan initialized.\r\n");
		}// Of if

		if (availableAlgorithms[FCM_CLUSTERING]) {
			fcm = new FCM(data, distanceMeasure);
			SimpleTools.processTrackingOutput("fcm initialized.\r\n");
		}// Of if

		if (availableAlgorithms[RANDOM_WALK_CLUSTERING]) {
			randomWalks = new RandomWalks(data, distanceMeasure);
			SimpleTools.processTrackingOutput("randomWalks initialized.\r\n");
		}// Of if

		if (availableAlgorithms[DP_CUTOFF_CLUSTERING]) {
			densityPeaksCutoff = new DensityPeaks(data, distanceMeasure, 0,
					adaptiveRatio, DensityPeaks.CUTOFF_KERNEL);
			SimpleTools
					.processTrackingOutput("densityPeaksCutoff initialized.\r\n");
		}// Of if

		SimpleTools
				.processTrackingOutput("ClusteringAlgorithmsBasedActiveLearning.initializePool() done.\r\n");
	}// Of initializePool

	/**
	 ************************* 
	 * Set available algorithms.
	 * 
	 * @param paraAvailableAlgorithms
	 *            The available algorithms.
	 ************************* 
	 */
	public void setAvailableAlgorithms(boolean[] paraAvailableAlgorithms) {
		availableAlgorithms = paraAvailableAlgorithms;
	}// Of setAvailableAlgorithms

	/**
	 ************************* 
	 * Get the algorithm win array.
	 * 
	 * @return The algorithm win array.
	 ************************* 
	 */
	public int[] getAlgorithmWinArray() {
		return algorithmWinArray;
	}// Of getAlgorithmWinArray

}// Of class ClusteringAlgorithmsBasedActiveLearning
