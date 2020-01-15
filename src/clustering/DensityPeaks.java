package clustering;

import java.util.Arrays;

import common.*;
import exception.*;
import weka.core.Instances;

/**
 * The density peaks clustering algorithms.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: The simple version finished. Kernels may be added in the future<br>
 * Written time: April 10, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public class DensityPeaks extends DensityClustering {
	/**
	 * Default max distance.
	 */
	public static final double MAX_DISTANCE_ROOT = 10000;

	/**
	 * The virtual maximal distance for the root.
	 */
	double maxDistance;

	/**
	 * The master of each instance.
	 */
	private int[] masters;

	/**
	 * The distance to master.
	 */
	double[] distancesToMaster;

	/**
	 ********************
	 * The constructor for independent running.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraDcScheme
	 *            The dc scheme. DIAMETER_FOR_DC or AVERAGE_FOR_DC.
	 * @param paraDcRatio
	 *            The ratio for radius computation.
	 * @param paraKernel
	 *            The kernel, cutoff or Gaussian.
	 ********************
	 */
	public DensityPeaks(String paraFilename, int paraDistanceMeasure, int paraDcScheme, double paraDcRatio,
			int paraKernel) {
		super(paraFilename, paraDistanceMeasure, paraDcScheme, paraDcRatio, paraKernel);
		initialize();
	}// Of the first constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraData
	 *            The data set.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraDcScheme
	 *            The dc scheme. DIAMETER_FOR_DC or AVERAGE_FOR_DC.
	 * @param paraDcRatio
	 *            The ratio for radius computation.
	 * @param paraKernel
	 *            The kernel, cutoff or Gaussian.
	 ********************
	 */
	public DensityPeaks(Instances paraData, DistanceMeasure paraDistanceMeasure, int paraDcScheme, double paraDcRatio,
			int paraKernel) {
		super(paraData, paraDistanceMeasure, paraDcScheme, paraDcRatio, paraKernel);
		initialize();
	}// Of the constructor

	/**
	 ********************
	 * Initialize.
	 ********************
	 */
	private void initialize() {
		maxDistance = MAX_DISTANCE_ROOT;
		balanceTwoBlocks = true;

		SimpleTools.processTrackingOutput("Computing masters ... ");
		computeMastersEfficiently();
		SimpleTools.processTrackingOutput("done.\r\n");
		SimpleTools.variableTrackingOutput("The masters are: " + Arrays.toString(masters));
		// computePriority();
	}// Of initialize

	/**
	 ************************* 
	 * Cluster the given block in k using density peaks. New roots are selected
	 * according to the representative.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return Two blocks
	 ************************* 
	 */
	public int[][] clusterInK(int[] paraBlock, int paraK) {
		// The density of each instance is computed and stored already.

		// Step 1. Compute the master tree.
		int tempRoot = computeMasters(paraBlock);

		// Step 3. Compute new (paraK - 1) roots for clustering
		// The indices 1 to paraK - 1 are useful
		int[] tempNewRoots = new int[paraK + 1];
		tempNewRoots[0] = tempRoot;
		double[] tempMultiplixes = new double[paraK + 1];
		Arrays.fill(tempMultiplixes, -1);
		tempMultiplixes[0] = Double.MAX_VALUE;

		double tempMultiplex;
		for (int i = 0; i < paraBlock.length; i++) {
			// Do not compare with the root.
			if (masters[i] == -1) {
				continue;
			} // Of if

			// Attention: Outliers have density 1.
			tempMultiplex = (densities[paraBlock[i]] - 0.99) * distancesToMaster[i];
			for (int j = paraK - 1;; j--) {
				if (tempMultiplixes[j] < tempMultiplex) {
					// Move forward the tail
					tempMultiplixes[j + 1] = tempMultiplixes[j];
					tempNewRoots[j + 1] = tempNewRoots[j];
				} else {
					// Insert here.
					tempMultiplixes[j + 1] = tempMultiplex;
					tempNewRoots[j + 1] = i;
					break;
				} // Of if
			} // Of for j
		} // Of for i

		// System.out.println("The roots are: " +
		// Arrays.toString(tempNewRoots));

		// Step 4. Now cluster in k.
		int[] tempClusterIndices = new int[paraBlock.length];
		Arrays.fill(tempClusterIndices, -1);
		// The roots
		for (int i = 0; i < paraK; i++) {
			tempClusterIndices[tempNewRoots[i]] = i;
		} // of

		for (int i = 0; i < tempClusterIndices.length; i++) {
			if (tempClusterIndices[i] != -1) {
				continue;
			} // Of if

			tempClusterIndices[i] = coincideWithMaster(masters[i], masters, tempClusterIndices);
		} // Of for i

		// Step 5. Obtain the blocks.
		int[] tempBlockSizes = new int[paraK];
		// int tempFirstBlockSize = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			tempBlockSizes[tempClusterIndices[i]]++;
		} // Of for i
		clusters = new int[paraK][];
		for (int i = 0; i < clusters.length; i++) {
			clusters[i] = new int[tempBlockSizes[i]];
		} // Of for i

		int[] tempBlockIndices = new int[paraK];
		int tempBlockIndex;
		for (int i = 0; i < paraBlock.length; i++) {
			// System.out.println("i = " + i
			// + ", tempClusterIndices[i] = " + tempClusterIndices[i]
			// + ", tempBlockIndices[tempClusterIndices[i]] = " +
			// tempBlockIndices[tempClusterIndices[i]]);
			tempBlockIndex = tempClusterIndices[i];
			clusters[tempClusterIndices[i]][tempBlockIndices[tempBlockIndex]] = paraBlock[i];
			tempBlockIndices[tempClusterIndices[i]]++;
		} // Of for i

		return clusters;
	}// Of clusterInK

	/**
	 ************************* 
	 * Cluster the given block in two using density peaks. May change later to
	 * obtain balanced clusters.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraQueriedArray
	 *            The queried instance within the block. The indices should be a
	 *            subset of [0 .. paraBlock.length - 1].
	 * @return Two blocks
	 ************************* 
	 */
	public int[][] clusterInTwo(int[] paraBlock, int[] paraQueriedArray) throws UnableToClusterInKException {
		SimpleTools.processTrackingOutput(
				"DensityPeaks.clusterInTwo(int[], int[]), " + paraQueriedArray.length + " queried\r\n");
		// Step 1. Check whether or not there are different labels.
		boolean tempHasDifferentLabels = false;
		int tempFirstLabel = (int) data.instance(paraBlock[paraQueriedArray[0]]).classValue();
		int tempCurrentLabel;
		for (int i = 1; i < paraQueriedArray.length; i++) {
			tempCurrentLabel = (int) data.instance(paraBlock[paraQueriedArray[i]]).classValue();
			if (tempCurrentLabel != tempFirstLabel) {
				tempHasDifferentLabels = true;
				break;
			} // Of if
		} // Of for i

		if (!tempHasDifferentLabels) {
			throw new UnableToClusterInKException(
					"Exception occurred in DensityPeaks.clusterInTwo(int[], int[]). " + " No different labels.");
		} // Of if

		// The density of each instance is computed and stored already.
		// Step 2. Compute the master tree.
		int tempRoot = computeMasters(paraBlock);
		
		// Step 3. The root and respective label.
		int[] tempNewRoots = new int[2];
		tempNewRoots[0] = tempRoot;
		tempNewRoots[1] = -1;
		int tempRootLabel = -1;
		// Is the root queried?
		boolean tempRootQueried = false;
		for (int i = 0; i < paraQueriedArray.length; i++) {
			if (tempRoot == paraQueriedArray[i]) {
				tempRootQueried = true;
				tempRootLabel = (int) data.instance(paraBlock[tempRoot]).classValue();
				break;
			} // Of if
		} // Of for i

		if (!tempRootQueried) {
			// Find its nearest neighbor for the pseudo-label.
			int tempNearestToRootLabeled = -1;
			double tempMinimalDistance = Double.MAX_VALUE;
			double tempDistance;
			for (int i = 0; i < paraQueriedArray.length; i++) {
				tempDistance = distanceMeasure.distance(paraBlock[tempRoot], paraBlock[paraQueriedArray[i]]);
				if (tempMinimalDistance > tempDistance) {
					tempMinimalDistance = tempDistance;
					tempNearestToRootLabeled = paraQueriedArray[i];
				} // Of if
			} // Of for i
			tempRootLabel = (int) data.instance(paraBlock[tempNearestToRootLabeled]).classValue();
		} // Of if

		// Step 4. Compute candidate new roots with different label and highest representative.
		int tempNumCandidates = 5;
		int[] tempCandidateRoots = new int[tempNumCandidates + 2];
		Arrays.fill(tempCandidateRoots, -1);
		double[] tempRepresentativeArray = new double[tempNumCandidates + 2];
		tempRepresentativeArray[0] = Double.MAX_VALUE;
		
		double tempMultiplex;
		for (int i = 0; i < paraQueriedArray.length; i++) {
			if ((int) data.instance(paraBlock[paraQueriedArray[i]]).classValue() == tempRootLabel) {
				continue;
			} // Of if

			tempMultiplex = (densities[paraBlock[paraQueriedArray[i]]] - 0.99) * distancesToMaster[paraQueriedArray[i]];
			for (int j = tempNumCandidates; ; j--) {
				if (tempRepresentativeArray[j] < tempMultiplex) {
					tempRepresentativeArray[j + 1] = tempRepresentativeArray[j];
					tempCandidateRoots[j + 1] = tempCandidateRoots[j];
				} else {
					tempRepresentativeArray[j + 1] = tempMultiplex;
					tempCandidateRoots[j + 1] = paraQueriedArray[i];
					break;
				} // Of if
			}//Of for j
		} // Of for i
		
		//Step 5. Try at most tempNumCandidates times to obtain balanced blocks.
		double tempBestBalanceFactor = -1;
		for (int i = 0; i < tempNumCandidates; i ++) {
			//Step 5.1 Set the second root.
			tempNewRoots[1] = tempCandidateRoots[i + 1];
			if (tempNewRoots[1] < 0) {
				break;
			}//Of if
			
			// Step 5.2 Now cluster in 2.
			int[] tempClusterIndices = new int[paraBlock.length];
			Arrays.fill(tempClusterIndices, -1);

			// The roots
			for (int j = 0; j < 2; j++) {
				tempClusterIndices[tempNewRoots[j]] = j;
			} // Of for i

			for (int j = 0; j < tempClusterIndices.length; j++) {
				if (tempClusterIndices[j] != -1) {
					continue;
				} // Of if

				tempClusterIndices[j] = coincideWithMaster(masters[j], masters, tempClusterIndices);
			} // Of for i

			int tempSecondLabel = (int) data.instance(paraBlock[tempNewRoots[1]]).classValue();
			if (tempSecondLabel == tempRootLabel) {
				System.out.println("The roots have the same label.");
				System.exit(0);
			} // Of if

			int[][] tempClusters = null;
			
			// Step 5.2 Obtain the blocks.
			try {
				tempClusters = blockInformationToBlocks(paraBlock, tempClusterIndices, 2);
			} catch (LessBlocksThanRequiredException ee) {
				System.out.println("Internal error occurred in DensityPeaks.clusterInTwo().\r\n" + ee);
				System.exit(0);
			} // Of try
			
			if (!balanceTwoBlocks) {
				//Balancing not required.
				clusters = tempClusters;
				break;
			}//Of if
			
			// Step 5.3 Are blocks balance?
			double tempBalanceFactor = getBalanceFactor(tempClusters);
			SimpleTools.processTrackingOutput("DensityPeaks.clusterInTwo() balancing " + i
					+ " with factor " + tempBalanceFactor + " ...\r\n");
			if (tempBestBalanceFactor < tempBalanceFactor) {
				tempBestBalanceFactor = tempBalanceFactor;
				clusters = tempClusters;
			}//Of if

			if (tempBalanceFactor > FINE_BALANCE_THRESHOLD) {
				break;
			}//Of if
		}//Of for iteration

		return clusters;
	}// Of clusterInTwo

	/**
	 ************************* 
	 * The block of a node should be same as its master
	 * @param paraIndex The index of the given node.
	 * @param paraMasters The master array indicating the master of each instance.
	 * @param paraClusterIndices Cluster indices of all instances.
	 * @return The cluster index of the current node.
	 ************************* 
	 */
	public int coincideWithMaster(int paraIndex, int[] paraMasters, int[] paraClusterIndices) {
		if (paraClusterIndices[paraIndex] == -1) {
			int tempMaster = paraMasters[paraIndex];
			paraClusterIndices[paraIndex] = coincideWithMaster(tempMaster, paraMasters, paraClusterIndices);
		} // Of if

		return paraClusterIndices[paraIndex];
	}// Of coincideWithMaster

	/**
	 ****************** 
	 * Find top-k critical instances.
	 * 
	 * @param paraK The required number of critical instances..
	 * @return The array of critical instances.
	 ****************** 
	 */
	int[] computeCriticalInstances(int paraK) {
		return computeCriticalInstances(wholeBlock, paraK);
	}// Of computeCriticalInstances

	/**
	 ****************** 
	 * Find top-k critical instances. The values are the indices in the original
	 * dataset.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The required number of critical instances.
	 * @return The array of critical instances.
	 ****************** 
	 */
	public int[] computeCriticalInstances(int[] paraBlock, int paraK) {
		//No more than the size of the block.
		if (paraK > paraBlock.length) {
			paraK = paraBlock.length;
		}//Of if
		
		// Initialize, recompute the master tree.
		// The aim is to recompute distancesToMaster[].
		computeMasters(paraBlock);

		SimpleTools.variableTrackingOutput("The given block is " + Arrays.toString(paraBlock));
		int[] tempIndices = new int[paraK + 2];
		Arrays.fill(tempIndices, -1);
		int tempNumInstances = paraBlock.length;
		double[] tempTopPriorities = new double[paraK + 2];
		Arrays.fill(tempTopPriorities, -1);
		tempTopPriorities[0] = Double.MAX_VALUE;

		double tempPriority;
		for (int i = 0; i < tempNumInstances; i++) {
			tempPriority = densities[paraBlock[i]] * distancesToMaster[i];
			for (int j = paraK;; j--) {
				if (tempPriority > tempTopPriorities[j]) {
					tempTopPriorities[j + 1] = tempTopPriorities[j];
					tempIndices[j + 1] = tempIndices[j];
				} else {
					tempTopPriorities[j + 1] = tempPriority;
					tempIndices[j + 1] = paraBlock[i];
					break;
				} // Of if
			} // Of for j
		} // Of for i
		SimpleTools.variableTrackingOutput("The top priorities are: " + Arrays.toString(tempTopPriorities));
		SimpleTools.variableTrackingOutput("The tempTopIndices are: " + Arrays.toString(tempIndices));

		int[] resultIndices = new int[paraK];
		for (int i = 0; i < resultIndices.length; i++) {
			resultIndices[i] = tempIndices[i + 1];
		} // Of for i

		return resultIndices;
	}// Of computeCriticalInstances

	/**
	 ****************** 
	 * Compute the masters efficiently.
	 ****************** 
	 */
	public void computeMastersEfficiently() {
		// Step 1. Check whether smaller blocks exist.
		if (smallerBlocks == null) {
			SimpleTools
					.processTrackingOutput("Smaller blocks not generated, compute masters slowly in DensityPeaks.\r\n");
			computeMasters(wholeBlock);
			return;
		} // Of if

		// Step 2. Initialize.
		masters = new int[numInstances];
		Arrays.fill(masters, -1);
		distancesToMaster = new double[numInstances];
		Arrays.fill(distancesToMaster, diameter);

		// Step 3. Compute masters of each instance in each block.
		boolean tempHasInnerBlockMaster;
		double tempDistance;
		for (int i = 0; i < smallerBlocks.length; i++) {
			SimpleTools.processTrackingOutput("block #" + i + ", ");
			for (int j = 0; j < smallerBlocks[i].length; j++) {
				// Step 3.1.1 Try to find a master in the same block.
				tempHasInnerBlockMaster = false;
				for (int k = 0; k < smallerBlocks[i].length; k++) {
					if (densities[smallerBlocks[i][k]] > densities[smallerBlocks[i][j]]) {
						tempHasInnerBlockMaster = true;
						tempDistance = distanceMeasure.distance(smallerBlocks[i][j], smallerBlocks[i][k]);
						if (distancesToMaster[smallerBlocks[i][j]] > tempDistance) {
							distancesToMaster[smallerBlocks[i][j]] = tempDistance;
							masters[smallerBlocks[i][j]] = smallerBlocks[i][k];
						} // Of if
					} // Of if
				} // Of for k

				if (tempHasInnerBlockMaster) {
					continue;
				} // Of if

				// SimpleTools.processTrackingOutput("#" + smallerBlocks[i][j] +
				// " is maximal with "
				// + densities[smallerBlocks[i][j]] + " \r\n");
				// Step 3.1.2 Find the master in the whole dataset
				for (int k = 0; k < numInstances; k++) {
					if (densities[k] > densities[smallerBlocks[i][j]]) {
						tempDistance = distanceMeasure.distance(smallerBlocks[i][j], k);
						if (distancesToMaster[smallerBlocks[i][j]] > tempDistance) {
							distancesToMaster[smallerBlocks[i][j]] = tempDistance;
							masters[smallerBlocks[i][j]] = k;
						} // Of if
					} // Of if
				} // Of for k
			} // Of for j
		} // Of for i

		SimpleTools.processTrackingOutput("\r\n");

		// Step 3. Only reserve one root.
		// The final root.
		int tempRoot = 0;
		for (int i = 0; i < numInstances; i++) {
			if (masters[i] == -1) {
				tempRoot = i;
				break;
			} // Of if
		} // Of for i

		// Others are not root.
		for (int i = tempRoot + 1; i < numInstances; i++) {
			if (masters[i] == -1) {
				masters[i] = tempRoot;
				distancesToMaster[i] = distanceMeasure.distance(i, tempRoot);
			} // Of if
		} // Of for i

		SimpleTools.variableTrackingOutput("The masters are: " + Arrays.toString(masters));
		SimpleTools.variableTrackingOutput("The distances to master are: " + Arrays.toString(distancesToMaster));

		SimpleTools.processTrackingOutput("computeMastersEfficiently() finished.\r\n");
	}// Of computeMastersEfficiently

	/**
	 ****************** 
	 * Compute the masters. At the same time, compute the distance to master.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @return The root of the master tree, indicated by the index within the
	 *         block.
	 ****************** 
	 */
	public int computeMasters(int[] paraBlock) {
		// Step 1. Initialize.
		int tempNumInstances = paraBlock.length;
		masters = new int[tempNumInstances];
		distancesToMaster = new double[tempNumInstances];

		// Step 2. Compute masters of each instance.
		double tempDistance;
		for (int i = 0; i < tempNumInstances; i++) {
			masters[i] = -1;
			distancesToMaster[i] = diameter;
			for (int j = 0; j < tempNumInstances; j++) {
				if (densities[paraBlock[j]] > densities[paraBlock[i]]) {
					tempDistance = distanceMeasure.distance(paraBlock[i], paraBlock[j]);
					if (distancesToMaster[i] > tempDistance) {
						distancesToMaster[i] = tempDistance;
						masters[i] = j;// **************
					} // Of if
				} // Of if
			} // Of for j
		} // Of for i

		// Step 3. Only reserve one root.
		// The final root.
		int resultRoot = 0;
		for (int i = 0; i < tempNumInstances; i++) {
			if (masters[i] == -1) {
				resultRoot = i;
				break;
			} // Of if
		} // Of for i

		// Others are not root.
		for (int i = resultRoot + 1; i < tempNumInstances; i++) {
			if (masters[i] == -1) {
				SimpleTools.variableTrackingOutput("Fixing " + i + " with density " + densities[i]);
				masters[i] = resultRoot;
				distancesToMaster[i] = distanceMeasure.distance(i, resultRoot);
			} // Of if
		} // Of for i

		SimpleTools.variableTrackingOutput("The masters are: " + Arrays.toString(masters));
		SimpleTools.variableTrackingOutput("The distances to master are: " + Arrays.toString(distancesToMaster));

		return resultRoot;
	}// Of computeMasters

	/**
	 ************************* 
	 * Test the method.
	 ************************* 
	 */
	public void testComputeDensityEfficiently() {
		System.out.println("testComputeDensityEfficiently, dc = " + dc);

		computeDensityEfficiently();
		System.out.println("If compute them efficiently, the densities are:\r\n" + Arrays.toString(densities));
	}// Of testComputeDensityEfficiently

	/**
	 ************************* 
	 * Test the method.
	 ************************* 
	 */
	public void testClusterInTwo() {
		// int[] tempBlock = { 1, 3, 49, 56, 88, 89, 99, 121, 123, 133 };
		int[] tempBlock = wholeBlock;
		int[] tempQueried = { 7, 60, 120 };
		int[][] tempPartition = null;

		try {
			tempPartition = clusterInTwo(tempBlock, tempQueried);
		} catch (UnableToClusterInKException ee) {
			System.out.println(ee);
		} // Of try

		SimpleTools.consoleOutput("With density peaks, the partition is: " + Arrays.deepToString(tempPartition));
	}// Of testClusterInTwo

	/**
	 ************************* 
	 * The main entrance.
	 * 
	 * @author Fan Min
	 * @param args The parameters.
	 ************************* 
	 */
	public static void main(String[] args) {
		SimpleTools.consoleOutput("Hello, densityPeaks.");
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/spiral_disorder.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		DensityPeaks densityPeaks = new DensityPeaks(tempFilename, DistanceMeasure.EUCLIDEAN, DIAMETER_FOR_DC, 0.1, 1);

		densityPeaks.testClusterInTwo();
		// densityPeaks.testClusterInK(2);

		double tempAccuracy = densityPeaks.computeAccuracy();
		System.out.println("The accuracy is: " + tempAccuracy);

		// densityPeaks.testComputeDensityEfficiently();
	}// Of main
}// Of class DensityPeaks
