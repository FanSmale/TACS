package clustering;

import java.util.Arrays;

import weka.core.Instances;
import common.DistanceMeasure;
import exception.LessBlocksThanRequiredException;
import exception.UnableToClusterInKException;
import matrix.*;

/**
 * The random walk clustering algorithms.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: The simple version finished. In some cases it cannot cluster the
 * block in two. Maybe we should fix it in the future. <br>
 * Written time: July 25, 2019. <br>
 * Last modify time: July 25, 2019.
 */

public class RandomWalks extends Clustering {
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
	public RandomWalks(String paraFilename, int paraDistanceMeasure) {
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
	public RandomWalks(Instances paraData, DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure);
	}// Of the second constructor

	/**
	 *********************
	 * The main algorithm.
	 * 
	 * @param paraFilename
	 *            The name of the decision table, or triple file.
	 * @param paraNumRounds
	 *            The rounds for random walk, each round update the weights,
	 *            however does not change the topology.
	 * @param paraK
	 *            The maximal times for matrix multiplex.
	 * @param paraMinNeighbors
	 *            For converting decision system into matrix only.
	 * @param paraCutThreshold
	 *            For final clustering from the result matrix. Links smaller
	 *            than the threshold will break.
	 *********************
	 *            public void randomWalk(String paraFilename, int paraNumRounds,
	 *            int paraK, int paraMinNeighbors, double paraCutThreshold) { //
	 *            Step 1. Read data CompressedMatrix tempMatrix = new
	 *            CompressedMatrix(paraFilename, paraMinNeighbors);
	 *            //System.out.println("The original matrix is: " + tempMatrix);
	 *            CompressedMatrix tempMultiplexion,
	 *            tempCombinedTransitionMatrix;
	 * 
	 *            // Step 2. Run a number of rounds to obtain new matrices for
	 *            (int i = 0; i < paraNumRounds; i++) { // Step 2.1 Compute
	 *            probability matrix CompressedMatrix tempProbabilityMatrix =
	 *            tempMatrix.computeTransitionProbabilities();
	 *            //System.out.println("\r\nThe probability matrix is:" +
	 *            tempProbabilityMatrix); // Make a copy tempMultiplexion = new
	 *            CompressedMatrix(tempProbabilityMatrix);
	 * 
	 *            // Step 2.2 Multiply and add // Reinitialize
	 *            tempCombinedTransitionMatrix = new
	 *            CompressedMatrix(tempProbabilityMatrix); for (int j = 2; j <=
	 *            paraK; j++) { //System.out.println("j = " + j);
	 *            tempMultiplexion = CompressedMatrix.multiply(tempMultiplexion,
	 *            tempProbabilityMatrix); tempCombinedTransitionMatrix =
	 *            CompressedMatrix.add(tempCombinedTransitionMatrix,
	 *            tempMultiplexion); } // Of for j
	 * 
	 *            //System.out.println("Find the error!" + tempMatrix);
	 * 
	 *            // Step 2.3 Distance between adjacent nodes for (int j = 0; j
	 *            < tempMatrix.matrix.length; j++) { Triple tempCurrentTriple =
	 *            tempMatrix.matrix[j].next; while (tempCurrentTriple != null) {
	 *            // Update the weight tempCurrentTriple.weight =
	 *            tempCombinedTransitionMatrix.neighborhoodSimilarity(j,
	 *            tempCurrentTriple.column, paraK);
	 * 
	 *            tempCurrentTriple = tempCurrentTriple.next; } // Of while } //
	 *            Of for i } // Of for i
	 * 
	 *            //System.out.println("The new matrix is:" + tempMatrix);
	 * 
	 *            // Step 3. Depth-first clustering and output
	 *            //tempMatrix.depthFirstClustering(paraCutThreshold);
	 * 
	 *            // Step 3'. Width-first clustering and output try {
	 *            tempMatrix.widthFirstClustering(paraCutThreshold); } catch
	 *            (Exception ee) { System.out.println("Error occurred in random
	 *            walk: " + ee); }//Of try }// Of randomWalk
	 */

	/**
	 *********************
	 * Cluster into k blocks.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraK
	 *            The number of sub-blocks.
	 *********************
	 * 
	 */
	public int[][] clusterInK(int[] paraBlock, int paraK) throws UnableToClusterInKException {
		int tempMinNeighbors = 50;
		int tempNumRounds = 3;
		double tempCutThreshold = 9.0;

		// Step 1. Reconstruct data and the matrix.
		Instances tempData = new Instances(dataHeader);
		for (int i = 0; i < paraBlock.length; i++) {
			tempData.add(data.instance(paraBlock[i]));
		} // Of for i
		CompressedMatrix tempMatrix = new CompressedMatrix(tempData, distanceMeasure, tempMinNeighbors);
		// System.out.println("The original matrix is: " + tempMatrix);
		CompressedMatrix tempMultiplexion, tempCombinedTransitionMatrix;

		// Step 2. Run a number of rounds to obtain new matrices
		for (int i = 0; i < tempNumRounds; i++) {
			// Step 2.1 Compute probability matrix
			CompressedMatrix tempProbabilityMatrix = tempMatrix.computeTransitionProbabilities();
			// System.out.println("\r\nThe probability matrix is:" +
			// tempProbabilityMatrix);
			// Make a copy
			tempMultiplexion = new CompressedMatrix(tempProbabilityMatrix);

			// Step 2.2 Multiply and add
			// Reinitialize
			tempCombinedTransitionMatrix = new CompressedMatrix(tempProbabilityMatrix);
			for (int j = 2; j <= paraK; j++) {
				tempMultiplexion = CompressedMatrix.multiply(tempMultiplexion, tempProbabilityMatrix);
				tempCombinedTransitionMatrix = CompressedMatrix.add(tempCombinedTransitionMatrix, tempMultiplexion);
			} // Of for j

			// System.out.println("Find the error!" + tempMatrix);

			// Step 2.3 Distance between adjacent nodes
			for (int j = 0; j < tempMatrix.matrix.length; j++) {
				Triple tempCurrentTriple = tempMatrix.matrix[j].next;
				while (tempCurrentTriple != null) {
					// Update the weight
					tempCurrentTriple.weight = tempCombinedTransitionMatrix.neighborhoodSimilarity(j,
							tempCurrentTriple.column, paraK);

					tempCurrentTriple = tempCurrentTriple.next;
				} // Of while
			} // Of for i
		} // Of for i

		// System.out.println("The new matrix is:" + tempMatrix);

		// Step 3. Depth-first clustering and output
		// tempMatrix.depthFirstClustering(paraCutThreshold);

		// Step 3'. Width-first clustering and output
		int[] tempBlockInformation = null;
		try {
			tempBlockInformation = tempMatrix.widthFirstClustering(tempCutThreshold);
			// System.out.println("tempBlockInformation = " +
			// Arrays.toString(tempBlockInformation));
		} catch (Exception ee) {
			System.out.println("Error occurred in random walk: " + ee);
		} // Of try

		int[][] resultBlocks = null;
		try {
			resultBlocks = blockInformationToBlocks(paraBlock, tempBlockInformation, paraK);
		} catch (LessBlocksThanRequiredException ee) {
			throw new UnableToClusterInKException(ee.toString(), paraK);
		} // Of try

		return resultBlocks;
	}// Of clusterInK

	public static void main(String args[]) {
		System.out.println("Let's randomly walk!");
		// KMeans tempMeans = new
		// KMeans("D:/workplace/randomwalk/data/iris.arff");
		// KMeans tempMeans = new
		// KMeans("D:/workspace/randomwalk/data/iris.arff");
		// Walk tempWalk = new Walk("D:/workspace/randomwalk/data/iris.arff");
		// int[] tempIntArray = {1, 2};

		// tempMeans.kMeans(3, KMeans.MANHATTAN);
		// tempMeans.kMeans(3, KMeans.EUCLIDEAN);
		// tempWalk.computeVkS(tempIntArray, 3);
		// double[][] tempMatrix = tempWalk.computeTransitionProbabilities();
		// double[][] tempTransition =
		// tempWalk.computeKStepTransitionProbabilities(100);
		// double[][] tempTransition =
		// tempWalk.computeAtMostKStepTransitionProbabilities(5);

		// double[][] tempNewGraph = tempWalk.ngSeparate(3);

		// System.out.println(Arrays.deepToString(tempMatrix));

		// System.out.println("The new graph is:\r\n" +
		// Arrays.deepToString(tempNewGraph));

		// CompressedSymmetricMatrix tempMatrix = new
		// CompressedSymmetricMatrix("D:/workspace/randomwalk/data/iris.arff",
		// 3);
		// CompressedSymmetricMatrix tempMatrix2 =
		// CompressedSymmetricMatrix.multiply(tempMatrix, tempMatrix);
		// CompressedSymmetricMatrix tempMatrix2 =
		// CompressedSymmetricMatrix.weightMatrixToTransitionProbabilityMatrix(tempMatrix);

		// System.out.println("The new matrix is: \r\n" + tempMatrix2);
		// System.out.println("The accuracy is: " + tempMeans.computePurity());

		// new
		// RandomWalk().randomWalk("D:/workspace/randomwalk/data/example21.arff",
		// 1, 3);

		//RandomWalks randomWalk = new RandomWalks("src/data/spiral.arff", DistanceMeasure.MANHATTAN);
		RandomWalks randomWalk = new RandomWalks("src/data/mushroom.arff", DistanceMeasure.EUCLIDEAN);
		int[][] resultBlocks = null;
		try {
			resultBlocks = randomWalk.clusterInK(2);
		} catch (UnableToClusterInKException ee) {
			System.out.println(ee);
		} // Of try

		System.out.print("Sizes: ");
		for (int i = 0; i < resultBlocks.length; i++) {
			System.out.print("" + resultBlocks[i].length + ", ");
		}//Of for i
		
		System.out.println("\r\nResult: " + Arrays.deepToString(resultBlocks));
	}// Of main
}// Of class RandomWalk
