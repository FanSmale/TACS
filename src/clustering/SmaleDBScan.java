package clustering;

import java.util.*;

import common.*;
import exception.UnableToClusterInKException;
import weka.core.*;

/**
 * The DBScan algorithm.
 * <p>
 * Author: <b>Fan Min</b>, <b>Shi-Ming Zhang</b> minfanphd@163.com,
 * minfan@swpu.edu.cn <br>
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

public class SmaleDBScan extends DensityClustering {
	/**
	 * The EPS ratio.
	 */
	public static final double DEFAULT_EPS_RATIO = 0.1;

	/**
	 * The EPS ratio.
	 */
	double epsRatio;

	/**
	 * The core threshold. Instances with density no smaller than the threshold
	 * are core.
	 */
	int coreThrehold;

	/**
	 * The search mark.
	 */
	boolean[] isVisitedALL;

	/**
	 * The EPS.
	 */
	double EPS;

	/**
	 * The distance matrix.
	 */
	double[][] disMatrix;

	/**
	 * The instances type. True for core, and false for otherwise.
	 */
	boolean[] isCoreArray;

	/**
	 * A boolean array indicating which instances have been visited.
	 */
	boolean[] isVisitedArray;

	/**
	 * An int array for the queue.
	 */
	int[] coreQueue;

	/**
	 * The cluster number of the instances.
	 */
	int[] clusterNumberArray;

	/**
	 * The density threshold of being a core.
	 */
	int coreDensityThreshold;

	/**
	 * The head of the queue.
	 */
	int head;

	/**
	 * The tail of the queue.
	 */
	int tail;

	/**
	 ********************
	 * The constructor for independent running.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraKernel
	 *            The kernel function.
	 ********************
	 */
	public SmaleDBScan(String paraFilename, int paraDistanceMeasure, int paraKernel) {
		super(paraFilename, paraDistanceMeasure, AVERAGE_FOR_DC, 0.2,
				paraKernel);
		epsRatio = DEFAULT_EPS_RATIO;
	}// Of the first constructor

	/**
	 ********************
	 * The constructor for independent running.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraEpsRatio
	 *            The EPS ratio.
	 * @param paraKernel
	 *            The kernel function.
	 ********************
	 */
	public SmaleDBScan(String paraFilename, int paraDistanceMeasure,
			double paraEpsRatio, int paraKernel) {
		super(paraFilename, paraDistanceMeasure, AVERAGE_FOR_DC, 0.1,
				paraKernel);
		SimpleTools.normalizeDecisionSystem(data);
		epsRatio = paraEpsRatio;
	}// Of the second constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData
	 *            The data set.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraKernel
	 *            The kernel function.
	 ********************
	 */
	public SmaleDBScan(Instances paraData, DistanceMeasure paraDistanceMeasure,
			int paraKernel) {
		super(paraData, paraDistanceMeasure, AVERAGE_FOR_DC, 0.2, paraKernel);
		epsRatio = DEFAULT_EPS_RATIO;
	}// Of the third constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData
	 *            The data set.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraEpsRatio
	 *            The EPS ratio.
	 * @param paraKernel
	 *            The given kernel function.
	 ********************
	 */
	public SmaleDBScan(Instances paraData, DistanceMeasure paraDistanceMeasure,
			double paraEpsRatio, int paraKernel) {
		super(paraData, paraDistanceMeasure, AVERAGE_FOR_DC, 0.2, paraKernel);
		epsRatio = paraEpsRatio;
	}// Of the fourth constructor

	/**
	 ********************
	 * The constructor for independent running.
	 * 
	 * @param paraCenter
	 *            Current search instance.
	 * @param paraBlock
	 *            The current data set.
	 * @param paraNowCluster
	 *            The current cluster set.
	 ********************
	 */
	void SearchNeighbor(int paraCenter, int[] paraBlock,
			ArrayList<Integer> paraNowCluster) {
		for (int i = 0; i < paraBlock.length; i++) {
			if (i == paraCenter) {
				continue;
			} // Of if

			if (disMatrix[paraCenter][i] < EPS && isVisitedALL[i] == false
					&& !isCoreArray[i]) {
				paraNowCluster.add(i);
				isVisitedALL[i] = true;

			} else if (disMatrix[paraCenter][i] < EPS
					&& isVisitedALL[i] == false && isCoreArray[i]) {
				paraNowCluster.add(i);
				isVisitedALL[i] = true;
				SearchNeighbor(i, paraBlock, paraNowCluster);
			} // Of else if
		} // Of for i
	}// Of SearchNeighbor

	/**
	 ********************
	 * Cluster using the default initial index array.
	 * 
	 * @return An array indicating the block index of each instance.
	 ********************
	 */
	int[] cluster() {
		//SimpleTools.consoleOutput("The data is: \r\n" + data);
		System.out.println("The data is: \r\n" + data);
		int[] tempInitialIndexArray = new int[data.numInstances()];
		for (int i = 0; i < tempInitialIndexArray.length; i++) {
			tempInitialIndexArray[i] = i;
		} // Of for i

		return clusterByQueue(tempInitialIndexArray);
	}// Of cluster

	/**
	 ********************
	 * Cluster in two using the default initial index array.
	 * 
	 * @return An array indicating the block index of each instance.
	 ********************
	 */
	public int[][] clusterInTwo() {
		SimpleTools.consoleOutput("The data is: \r\n" + data);
		int[] tempInitialIndexArray = new int[data.numInstances()];
		for (int i = 0; i < tempInitialIndexArray.length; i++) {
			tempInitialIndexArray[i] = i;
		} // Of for i

		return clusterInTwo(tempInitialIndexArray);
	}// Of cluster

	/**
	 ********************
	 * Set the cluster number of instances that can be reached by the core.
	 * 
	 * @param paraCore
	 *            The given core.
	 * @param paraNumber
	 *            The cluster number.
	 ********************
	 */
	private void setClusterNumberUsingCore(int[] paraBlock, int paraCore,
			int paraNumber) {
		// Step 3.2 Initialize the queue.
		head = 0;
		coreQueue[0] = paraCore;
		clusterNumberArray[paraCore] = paraNumber;
		isVisitedArray[paraCore] = true;
		tail = 1;
		int tempCurrentCore;

		while (head < tail) {
			// Step 3.3.1 Take out the head. Dequeue.
			tempCurrentCore = coreQueue[head];
			head++;

			// Step 3.3.2 Put the neighbors to the tail. Enqueue.
			for (int i = 0; i < paraBlock.length; i++) {
				if (i == tempCurrentCore) {
					continue;
				} // Of if

				if (isVisitedArray[i]) {
					continue;
				} // Of if

				if (distanceMeasure.distance(paraBlock[tempCurrentCore],
						paraBlock[i]) <= dc) {
					clusterNumberArray[i] = paraNumber;
					isVisitedArray[i] = true;
					// System.out.println("The core queue is: "
					// + Arrays.toString(coreQueue));

					if (isCoreArray[i]) {
						coreQueue[tail] = i;
						tail++;
					} // Of if
				} // Of if
			} // Of for j
		} // Of while
	}// Of setClusterNumberUsingCore

	/**
	 ********************
	 * Cluster using a queue
	 * 
	 * @param tempInitialIndexArray
	 *            The given
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 ********************
	 */
	int[] clusterByQueue(int[] paraBlock) {
		setAdaptiveDc(paraBlock, 0.1);

		// Step 1. Assign space for all arrays and initialize.
		isVisitedArray = new boolean[paraBlock.length];
		coreQueue = new int[paraBlock.length + 1];
		clusterNumberArray = new int[paraBlock.length];
		Arrays.fill(clusterNumberArray, -1);
		coreDensityThreshold = (int) Math.sqrt(paraBlock.length) + 1;

		System.out.println("coreDensityThreshold = "
						+ coreDensityThreshold);
		//SimpleTools.consoleOutput("coreDensityThreshold = "
		//		+ coreDensityThreshold);

		// Step 2. Which ones are core?
		isCoreArray = new boolean[paraBlock.length];
		int tempNumNeighbors;
		for (int i = 0; i < paraBlock.length; i++) {
			tempNumNeighbors = 0;
			for (int j = 0; j < paraBlock.length; j++) {
				if (distanceMeasure.distance(paraBlock[i], paraBlock[j]) <= dc) {
					tempNumNeighbors++;
					if (tempNumNeighbors >= coreDensityThreshold) {
						isCoreArray[i] = true;
						System.out.println("Core instance: " + i);
						break;
					} // Of if
				} // Of if
			} // Of for j
		} // Of for i

		// Step 3. Initialize the queue to include only one core.
		int tempCurrentBlockNumber = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			// Step 3.1 Handle other cases.
			// Already visited.
			if (isVisitedArray[i]) {
				continue;
			} // Of if

			// Not a core.
			if (!isCoreArray[i]) {
				continue;
			} // Of if

			tempCurrentBlockNumber++;

			// Step 3.2 Set the cluster number of instances reached by instance
			// paraBlock[i].
			setClusterNumberUsingCore(paraBlock, i, tempCurrentBlockNumber);
		} // Of for i

		// System.out
		// .println("The core array is: " + Arrays.toString(isCoreArray));
		// System.out.println("The core queue is: " +
		// Arrays.toString(coreQueue));
		// System.out.println("The visited array is: "
		// + Arrays.toString(isVisitedArray));
		
		//SimpleTools.consoleOutput("There are " + (tempCurrentBlockNumber + 1)
		//		+ " blocks.");
		System.out.println("There are " + (tempCurrentBlockNumber + 1)
				+ " blocks.");
		return clusterNumberArray;
	}// Of clusterByQueue

	/**
	 ************************* 
	 * Cluster the given block in two using DBScan.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return Two blocks
	 ************************* 
	 */
	public int[][] clusterInTwo(int[] paraBlock) {
		double tempDcRatio = DEFAULT_DC_RATIO;
		setAdaptiveDc(paraBlock, tempDcRatio);

		// Step 1. Assign space for all arrays and initialize.
		isVisitedArray = new boolean[paraBlock.length];
		coreQueue = new int[paraBlock.length];
		clusterNumberArray = new int[paraBlock.length];
		Arrays.fill(clusterNumberArray, -1);
		coreDensityThreshold = (int) Math.sqrt(paraBlock.length) + 1;

		// Step 2. Which ones are core?
		isCoreArray = new boolean[paraBlock.length];
		int tempNumNeighbors;

		boolean tempHasCore = false;
		while (tempDcRatio < 1) {
			for (int i = 0; i < paraBlock.length; i++) {
				tempNumNeighbors = 0;
				for (int j = 0; j < paraBlock.length; j++) {
					if (distanceMeasure.distance(paraBlock[i], paraBlock[j]) <= dc) {
						tempNumNeighbors++;
						if (tempNumNeighbors >= coreDensityThreshold) {
							isCoreArray[i] = true;
							tempHasCore = true;
							break;
						} // Of if
					} // Of if
				} // Of for j
			} // Of for i

			if (tempHasCore) {
				System.out.println("tempDcRatio = " + tempDcRatio);
				break;
			} // Of if

			// Adjust tempDcRatio and respective dc
			tempDcRatio *= 1.5;
			setAdaptiveDc(paraBlock, tempDcRatio);
		} // Of while

		// Never find a core? Why?
		if (!tempHasCore) {
			System.out.println("Warning: The block is: "
					+ Arrays.toString(paraBlock));
			System.out.println("The core density threhold "
					+ coreDensityThreshold + " can never be met.");
			// Fail.
			return null;
		} // Of if

		// Step 3. Handle the first queue.
		// Step 3.1 Find the first core
		int tempFirstCore = -1;
		for (int i = 0; i < paraBlock.length; i++) {
			// Not a core.
			if (!isCoreArray[i]) {
				continue;
			} else {
				tempFirstCore = i;
				break;
			} // Of if
		} // Of for i

		if (tempFirstCore == -1) {
			System.out
					.println("Error occurred in DBScan.clusterInTwo(int[]). There is no core.");
			System.out.println(Arrays.toString(isCoreArray));
			System.exit(0);
		} // Of if

		setClusterNumberUsingCore(paraBlock, tempFirstCore, 0);

		// Step 4. Handle the second queue.
		// Step 4.1 Find the second core, it is the farthest core from the first
		// one.
		double tempMaxDistance = -1;
		int tempSecondCore = -1;
		double tempDistance;
		for (int i = 0; i < paraBlock.length; i++) {
			if (!isCoreArray[i]) {
				continue;
			} // Of if

			if (isVisitedArray[i]) {
				continue;
			} // Of if

			tempDistance = distanceMeasure.distance(paraBlock[tempFirstCore],
					paraBlock[i]);
			if (tempDistance > tempMaxDistance) {
				tempMaxDistance = tempDistance;
				tempSecondCore = i;
			} // Of if
		} // Of for i

		if (tempSecondCore == -1) {
			SimpleTools
					.consoleOutput("Error occurred in DBSCan! Cannot find the next core.");
			SimpleTools.consoleOutput("The core array is: "
					+ Arrays.toString(isCoreArray));
			SimpleTools.consoleOutput("The visited array is: "
					+ Arrays.toString(isVisitedArray));

			// The algorithm fails.
			return null;
		} // Of if

		setClusterNumberUsingCore(paraBlock, tempSecondCore, 1);

		// Step 5. Assign cluster number for other instances
		// Now use the simplest strategy, i.e., only compare the distances to
		// the first and the second cores.
		double tempDistanceToFirst, tempDistanceToSecond;
		for (int i = 0; i < paraBlock.length; i++) {
			if (isVisitedArray[i]) {
				continue;
			} // Of if

			// Find nearest neighbors
			tempDistanceToFirst = distanceMeasure.distance(paraBlock[i],
					paraBlock[tempFirstCore]);
			tempDistanceToSecond = distanceMeasure.distance(paraBlock[i],
					paraBlock[tempSecondCore]);

			if (tempDistanceToFirst <= tempDistanceToSecond) {
				clusterNumberArray[i] = 0;
			} else {
				clusterNumberArray[i] = 1;
			} // Of if
		} // Of for i

		// Step 6. Int array to int matrix.
		int tempFirstBlockSize = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (clusterNumberArray[i] == 0) {
				tempFirstBlockSize++;
			} // Of if
		} // Of for i

		int[][] resultBlocks = new int[2][];
		resultBlocks[0] = new int[tempFirstBlockSize];
		resultBlocks[1] = new int[paraBlock.length - tempFirstBlockSize];

		int tempFirstBlockIndex = 0;
		int tempSecondBlockIndex = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (clusterNumberArray[i] == 0) {
				resultBlocks[0][tempFirstBlockIndex] = paraBlock[i];
				tempFirstBlockIndex++;
			} else {
				resultBlocks[1][tempSecondBlockIndex] = paraBlock[i];
				tempSecondBlockIndex++;
			} // Of for i
		} // Of for i

		// System.out
		// .println("The core array is: " + Arrays.toString(isCoreArray));
		// System.out.println("The core queue is: " +
		// Arrays.toString(coreQueue));
		// System.out.println("The visited array is: "
		// + Arrays.toString(isVisitedArray));
		SimpleTools.consoleOutput("The result blocks are: "
				+ Arrays.deepToString(resultBlocks));
		return resultBlocks;
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
	public int[][] clusterInK(int[] paraBlock, int paraK) {
		return null;
	}// Of clusterInK

	/**
	 ********************
	 * Set the eps ratio.
	 * 
	 * @param paraEpsRasio
	 *            The given ratio.
	 ********************
	 */
	void setEpsRatio(double paraEpsRasio) {
		epsRatio = paraEpsRasio;
	}// Of setEPSRatio

	/**
	 ************************* 
	 * The main entrance.
	 * 
	 * @author Fan Min
	 * @param args
	 *            The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		System.out.println("Hello, DBScan.");
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/real.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			System.out.println("The filename is: " + tempFilename);
		} // Of if

		SmaleDBScan tempDBScan = new SmaleDBScan(tempFilename,
				DistanceMeasure.EUCLIDEAN, 0.5, 0);
		Common.runtimes = 0;
		Common.startTime = new Date().getTime();
		int[] ResultArray = tempDBScan.cluster();
		
		Common.endTime = new Date().getTime();
		System.out.println("Final results: " + Arrays.toString(ResultArray));
		System.out.println("The runtime is: " + Common.runtimes);
		System.out.println("It is: " + (Common.endTime - Common.startTime)
				+ "ms.");

		int[][] tempBlocks = tempDBScan.clusterInTwo();
		System.out.println("Cluster in two. The results are: " + Arrays.deepToString(tempBlocks));
		
	}// Of main
}// Of SmaleDBScan
