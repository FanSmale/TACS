package clustering;

import common.*;
import exception.UnableToClusterInKException;
import weka.core.Instances;

/**
 * The super class of any means clustering algorithms, such as kMeans and FCM.
 * It help setting the initial centers.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Almost finished, further revision is possible.<br>
 * Written time: July 10, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public abstract class MeansClustering extends Clustering {
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
	public MeansClustering(String paraFilename, int paraDistanceMeasure) {
		super(paraFilename, paraDistanceMeasure);
	}// Of the first constructor

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
	public MeansClustering(Instances paraData, DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure);
	}// Of the second constructor

	/**
	 ************************* 
	 * Cluster the given block in two using kMeans, the centers are the first
	 * and the last elements.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return Two blocks
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ************************* 
	 */
	public int[][] clusterInTwo(int[] paraBlock) throws UnableToClusterInKException {
		// Step 1. Select two initial points
		double[][] tempCenters = getSemiMaximalDistancePair(paraBlock);

		return clusterInK(paraBlock, tempCenters);
	}// Of clusterInTwo

	/**
	 ************************* 
	 * Assign a number of randomly selected instances to centers.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The number of clusters.
	 * @return Clusters
	 ************************* 
	 */
	public double[][] getRandomCenters(int[] paraBlock, int paraK) {
		int[] tempArray = SimpleTools.getRandomOrder(paraBlock.length);
		double[][] resultCenters = new double[paraK][numConditions];

		for (int i = 0; i < paraK; i++) {
			for (int j = 0; j < numConditions; j++) {
				resultCenters[i][j] = data.instance(paraBlock[tempArray[i]]).value(j);
			} // Of for j
		} // Of for i

		return resultCenters;
	}// Of getRandomCenters

	/**
	 ************************* 
	 * Cluster the given block in using kMeans.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The number of clusters.
	 * @return Clusters
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ************************* 
	 */
	public int[][] clusterInK(int[] paraBlock, int paraK) throws UnableToClusterInKException {
		double[][] tempCenters = getRandomCenters(paraBlock, paraK);

		return clusterInK(paraBlock, tempCenters);
	}// Of clusterInK

	/**
	 ************************* 
	 * Cluster the given block in using kMeans.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraCenters
	 *            The given centers.
	 * @return Clusters
	 * @throws UnableToClusterInKException
	 *             If fails to cluster.
	 ************************* 
	 */
	public abstract int[][] clusterInK(int[] paraBlock, double[][] paraCenters) throws UnableToClusterInKException;

}// Of class MeansClustering
