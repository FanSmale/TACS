package clustering;

import java.util.Arrays;

import common.DistanceMeasure;
import common.IntegerNode;
import common.SimpleTools;
import exception.UnableToClusterInKException;
import weka.core.Instances;

/**
 * The hierarchical clustering algorithms.
 * <p>
 * Author: <b>Fan Min</b>, <b>Shi-Ming Zhang</b> minfanphd@163.com,
 * minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: The simple version finished. clusterInK not implemented yet.<br>
 * Written time: April 10, 2019. <br>
 * Last modify time: July 28, 2019.
 */

public class SmaleHierarchical extends Clustering {

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
	public SmaleHierarchical(String paraFilename, int paraDistanceMeasure) {
		super(paraFilename, paraDistanceMeasure);
	}// Of the first constructor

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
	public SmaleHierarchical(Instances paraData,
			DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure);
	}// Of the constructor

	/**
	 ************************* 
	 * Cluster the given block in k using the hierarchical clustering
	 * algorithm. Each time merge the two blocks in the same level.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The number of clusters.
	 * @return A partition of the block.
	 * @throws UnableToClusterInKException
	 *             If the data cannot be clustered in k.
	 ************************* 
	 */
	public int[][] clusterInK(int[] paraBlock, int paraK)
			throws UnableToClusterInKException {
		// Step 1. Initialize.
		int tempBlockLength = paraBlock.length;
		IntegerNode[] tempNodeArray = new IntegerNode[tempBlockLength];
		for (int i = 0; i < tempBlockLength; i++) {
			tempNodeArray[i] = new IntegerNode(i);
		} // Of for i
		boolean[] tempHead = new boolean[tempBlockLength];
		Arrays.fill(tempHead, true);
		int tempNumHeads = tempBlockLength;
		boolean[] tempCurrentLevelAvailable = new boolean[tempBlockLength];

		// Step 2. Merge a few levels until the current level has exactly paraK
		// blocks
		double tempMinDistance;
		double tempDistance;
		int tempFirstClusterHead = -1;
		int tempSecondClusterHead = -1;
		IntegerNode tempTail;
		
		// Handle the current level.
		while (tempNumHeads > paraK) {
			// System.out.println("New level");
			// Step 2.1 Initialize. All heads are available.
			for (int i = 0; i < tempCurrentLevelAvailable.length; i++) {
				tempCurrentLevelAvailable[i] = tempHead[i];
			} // Of for i

			// Step 2.2 Merge tempNumHeads/2 times
			for (int i = 0; i < tempNumHeads / 2; i++) {
				// Step 2.2.1 Initialize
				tempMinDistance = Double.MAX_VALUE;

				// Step 2.2.2 Find the two clusters to be merged.
				tempFirstClusterHead = -1;
				for (int j = 0; j < tempBlockLength; j++) {
					if (tempCurrentLevelAvailable[j]) {
						tempFirstClusterHead = j;
						break;
					} // Of if
				} // Of for j

				tempSecondClusterHead = -1;
				for (int j = tempFirstClusterHead + 1; j < tempBlockLength; j++) {
					if (!tempCurrentLevelAvailable[j]) {
						continue;
					} // Of if

					tempDistance = clusterDistance(paraBlock,
							tempNodeArray[tempFirstClusterHead],
							tempNodeArray[j]);
					if (tempDistance < tempMinDistance + 1e-6) {
						tempSecondClusterHead = j;
						tempMinDistance = tempDistance;
						// System.out.println("The distance between block " + j
						// + " and " + k + " is: " + tempDistance);
					} // Of if
				} // Of for j

				// Step 2.2.3 Now merge them.
				// Search the tail of the first cluster.
				tempTail = tempNodeArray[tempFirstClusterHead];
				while (tempTail.next != null) {
					tempTail = tempTail.next;
				} // Of while

				// Step 2.3 Link to the head of the second cluster.
				tempTail.next = tempNodeArray[tempSecondClusterHead];

				// Step 2.2.4 Set respective indicators
				tempHead[tempSecondClusterHead] = false;
				tempCurrentLevelAvailable[tempFirstClusterHead] = false;
				tempCurrentLevelAvailable[tempSecondClusterHead] = false;
			} // Of for i

			tempNumHeads -= tempNumHeads / 2;
		} // Of while

		// Step 3. Construct the block information array.
		int[] tempBlockInformationArray = new int[paraBlock.length];
		int tempClusterNumber = 0;
		IntegerNode tempNode = null;
		for (int i = 0; i < tempNodeArray.length; i++) {
			if (!tempHead[i]) {
				continue;
			} // Of if

			tempNode = tempNodeArray[i];
			while (tempNode != null) {
				tempBlockInformationArray[tempNode.value] = tempClusterNumber;
				tempNode = tempNode.next;
			} // Of while

			tempClusterNumber++;
		} // Of for i

		int[][] resultBlocks = null;
		try {
			resultBlocks = blockInformationToBlocks(paraBlock,
					tempBlockInformationArray, paraK);
		} catch (Exception ee) {
			throw new UnableToClusterInKException("Hierarchical.clusterInK(): "
					+ ee.toString());
		} // Of try

		return resultBlocks;
	}// Of clusterInK	
	
	/**
	 ************************* 
	 * Cluster the given block in k using the hierarchical clustering
	 * algorithm. Each time merge the two blocks in the same level.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The number of clusters.
	 * @return A partition of the block.
	 * @throws UnableToClusterInKException
	 *             If the data cannot be clustered in k.
	 ************************* 
	 */
	public int[][] clusterInKLevelwise(int[] paraBlock, int paraK)
			throws UnableToClusterInKException {
		// Step 1. Initialize.
		int tempBlockLength = paraBlock.length;
		IntegerNode[] tempNodeArray = new IntegerNode[tempBlockLength];
		for (int i = 0; i < tempBlockLength; i++) {
			tempNodeArray[i] = new IntegerNode(i);
		} // Of for i
		boolean[] tempHead = new boolean[tempBlockLength];
		Arrays.fill(tempHead, true);
		int tempNumHeads = tempBlockLength;
		boolean[] tempCurrentLevelAvailable = new boolean[tempBlockLength];

		// Step 2. Merge a few levels until the current level has exactly paraK
		// blocks
		double tempMinDistance;
		double tempDistance;
		int tempFirstClusterHead = -1;
		int tempSecondClusterHead = -1;
		IntegerNode tempTail;
		
		// Handle the current level.
		while (tempNumHeads > paraK) {
			// System.out.println("New level");
			// Step 2.1 Initialize. All heads are available.
			for (int i = 0; i < tempCurrentLevelAvailable.length; i++) {
				tempCurrentLevelAvailable[i] = tempHead[i];
			} // Of for i

			// Step 2.2 Merge tempNumHeads/2 times
			for (int i = 0; i < tempNumHeads / 2; i++) {
				// Step 2.2.1 Initialize
				tempMinDistance = Double.MAX_VALUE;

				// Step 2.2.2 Find the two clusters to be merged.
				tempFirstClusterHead = -1;
				for (int j = 0; j < tempBlockLength; j++) {
					if (tempCurrentLevelAvailable[j]) {
						tempFirstClusterHead = j;
						break;
					} // Of if
				} // Of for j

				tempSecondClusterHead = -1;
				for (int j = tempFirstClusterHead + 1; j < tempBlockLength; j++) {
					if (!tempCurrentLevelAvailable[j]) {
						continue;
					} // Of if

					tempDistance = clusterDistance(paraBlock,
							tempNodeArray[tempFirstClusterHead],
							tempNodeArray[j]);
					if (tempDistance < tempMinDistance + 1e-6) {
						tempSecondClusterHead = j;
						tempMinDistance = tempDistance;
						// System.out.println("The distance between block " + j
						// + " and " + k + " is: " + tempDistance);
					} // Of if
				} // Of for j

				// Step 2.2.3 Now merge them.
				// Search the tail of the first cluster.
				tempTail = tempNodeArray[tempFirstClusterHead];
				while (tempTail.next != null) {
					tempTail = tempTail.next;
				} // Of while

				// Step 2.3 Link to the head of the second cluster.
				tempTail.next = tempNodeArray[tempSecondClusterHead];

				// Step 2.2.4 Set respective indicators
				tempHead[tempSecondClusterHead] = false;
				tempCurrentLevelAvailable[tempFirstClusterHead] = false;
				tempCurrentLevelAvailable[tempSecondClusterHead] = false;
			} // Of for i

			tempNumHeads -= tempNumHeads / 2;
		} // Of while

		// Step 3. Construct the block information array.
		int[] tempBlockInformationArray = new int[paraBlock.length];
		int tempClusterNumber = 0;
		IntegerNode tempNode = null;
		for (int i = 0; i < tempNodeArray.length; i++) {
			if (!tempHead[i]) {
				continue;
			} // Of if

			tempNode = tempNodeArray[i];
			while (tempNode != null) {
				tempBlockInformationArray[tempNode.value] = tempClusterNumber;
				tempNode = tempNode.next;
			} // Of while

			tempClusterNumber++;
		} // Of for i

		int[][] resultBlocks = null;
		try {
			resultBlocks = blockInformationToBlocks(paraBlock,
					tempBlockInformationArray, paraK);
		} catch (Exception ee) {
			throw new UnableToClusterInKException("Hierarchical.clusterInK(): "
					+ ee.toString());
		} // Of try

		return resultBlocks;
	}// Of clusterInKLevelwise

	/**
	 ************************* 
	 * Compute the distance between two cluster centers.
	 * 
	 * @param paraFirstCluster
	 *            The first cluster indicated by a reference of IntegerNode.
	 * @param paraSecondCluster
	 *            The second cluster indicated by a reference of IntegerNode.
	 ************************* 
	 */
	private double clusterDistance(int[] paraBlock,
			IntegerNode paraFirstCluster, IntegerNode paraSecondCluster) {
		// Step 1. Compute centers.
		double[] tempFirstCenter = new double[data.numAttributes() - 1];
		double[] tempSecondCenter = new double[data.numAttributes() - 1];

		// Step 1.1 First center
		int tempCounter = 0;
		while (paraFirstCluster != null) {
			for (int i = 0; i < data.numAttributes() - 1; i++) {
				tempFirstCenter[i] += data.instance(
						paraBlock[paraFirstCluster.value]).value(i);
			} // Of for i
			tempCounter++;
			paraFirstCluster = paraFirstCluster.next;
		} // Of while
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempFirstCenter[i] /= tempCounter;
		} // Of for i

		// Step 1.2 Second center
		tempCounter = 0;
		while (paraSecondCluster != null) {
			for (int i = 0; i < data.numAttributes() - 1; i++) {
				tempSecondCenter[i] += data.instance(
						paraBlock[paraSecondCluster.value]).value(i);
			} // Of for i
			tempCounter++;
			paraSecondCluster = paraSecondCluster.next;
		} // Of while
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempSecondCenter[i] /= tempCounter;
		} // Of for i

		return distanceMeasure.distance(tempFirstCenter, tempSecondCenter);
	}// Of clusterDistance

	/**
	 ************************* 
	 * Test the ClusterInTwo method.
	 ************************* 
	 */
	public void testClusterInTwo() {
		int[] tempBlock = wholeBlock;
		// int[] tempBlock = { 1, 3, 49, 56, 88, 89, 99, 121, 123, 133 };
		// int[] tempBlock = {1, 3, 88, 89, 99, 121, 123, 133};
		// int[] tempBlock = {1, 88, 89, 99, 123, 133};

		// int[] tempBlock = {1, 3, 49, 56, 88, 89, 99};

		System.out.println("The original data is:");
		for (int i = 0; i < tempBlock.length; i++) {
			for (int j = 0; j < numConditions; j++) {
				System.out.print(" " + data.instance(tempBlock[i]).value(j));
			} // Of for j
			System.out.println("\r\n");
		} // Of for i

		System.out.println("Before clustering.");

		// int[][] tempPartition = clusterInTwo(tempBlock);
		int[][] tempPartition = null;
		try {
			tempPartition = clusterInTwo(tempBlock);
		} catch (Exception ee) {
			System.out.println(ee);
		} // Of try
			// int[][] tempPartition = clusterInK(tempBlock, 3);
		System.out.println("After clustering.");

		System.out.println("With hierachical, the partition is: ");
		System.out.println(Arrays.deepToString(tempPartition));
	}// Of testClusterInTwo

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
		SimpleTools.consoleOutput("Hello, hierarchical.");
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "E:/workplace/cenal/src/data/iris.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		SmaleHierarchical hierarchical = new SmaleHierarchical(tempFilename,
				DistanceMeasure.EUCLIDEAN);

		hierarchical.testClusterInTwo();
	}// Of main
}// Of class Hierarchical
