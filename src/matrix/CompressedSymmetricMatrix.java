package matrix;

import common.DistanceMeasure;
import weka.core.Instances;

public class CompressedSymmetricMatrix {
	/**
	 * The data.
	 */
	Instances data;

	/**
	 * The number instances.
	 */
	int numInstances;

	/**
	 * The matrix.
	 */
	Triple[] matrix;

	/**
	 * The distance measure.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 * How many nodes have been inserted to the list.
	 */
	int totalInserted;

	public static final int MAX_WEIGHT = 100000;

	/**
	 *********************
	 * The constructor. Invoke the constructor of the superclass directly.
	 * 
	 * @param paraData
	 *            The data.
	 * @param paraDistanceMeasure
	 *            The distance measure.
	 * @param paraK
	 *            The number of neighbors.
	 *********************
	 */
	public CompressedSymmetricMatrix(Instances paraData, DistanceMeasure paraDistanceMeasure, int paraK) {
		data = paraData;
		numInstances = data.numInstances();
		totalInserted = 0;

		// Initialize matrix
		matrix = new Triple[numInstances];
		for (int i = 0; i < matrix.length; i++) {
			matrix[i] = new Triple();
		} // Of for i

		// kNearestNeighbors(0, 3);
		// Scan the data to construct the matrix
		for (int i = 0; i < matrix.length; i++) {
			matrix[i] = kNearestNeighbors(i, paraK);
		} // Of for i

		System.out.println(this);
		symmetrize();
		System.out.println("After symmtrize\r\n.The vector is: " + this);

		System.out.println("Total inserted: " + totalInserted);
	}// of the first constructor

	/**
	 *********************
	 * The constructor. Construct an empty matrix with the given size.
	 * @param paraNumNodes The number of nodes, also the number of triples.
	 *********************
	 */
	public CompressedSymmetricMatrix(int paraNumNodes) {
		totalInserted = 0;

		// Initialize matrix
		matrix = new Triple[paraNumNodes];
		for (int i = 0; i < matrix.length; i++) {
			matrix[i] = new Triple();
		} // Of for i
	}// Of the second constructor

	/**
	 *********************
	 * Compute the k-nearest neighbors of the node.
	 * @param paraNode The given node.
	 * @param paraK k.
	 *********************
	 */
	Triple kNearestNeighbors(int paraNode, int paraK) {
		Triple resultHeader = new Triple();
		Triple tempReference = resultHeader;

		int[] tempIndices = new int[paraK + 1];
		double[] tempWeights = new double[paraK + 1];
		double tempSimilarity = 0;

		for (int i = 0; i < data.numInstances(); i++) {
			if (i == paraNode) {
				continue;
			} // Of if

			double tempDistance = distanceMeasure.distance(paraNode, i);
			if (tempDistance < 1e-10) {
				tempSimilarity = MAX_WEIGHT;
			} else {
				tempSimilarity = 1.0 / tempDistance;
			} // Of else
				// System.out.println("tempSimilarity = " + tempSimilarity);
			for (int j = 0; j < paraK; j++) {
				if (tempSimilarity > tempWeights[j]) {
					// Move the tail
					for (int k = paraK; k > j; k--) {
						tempIndices[k] = tempIndices[k - 1];
						tempWeights[k] = tempWeights[k - 1];
					} // Of for k

					// Now insert
					tempIndices[j] = i;
					tempWeights[j] = tempSimilarity;

					break;
				} // Of if
			} // Of for j
		} // Of for i

		// System.out.println("Indices: " + Arrays.toString(tempIndices) + ",
		// weights " + Arrays.toString(tempWeights));

		boolean[] tempProcessed = new boolean[paraK];
		int tempMinimalIndex = Integer.MAX_VALUE;
		double tempWeight = -1;
		int tempIndexInInicesArray = -1;
		// Search the whole array
		for (int i = 0; i < paraK; i++) {
			tempMinimalIndex = Integer.MAX_VALUE;
			// Find the minimal index
			for (int j = 0; j < paraK; j++) {
				if (tempProcessed[j]) {
					continue;
				} // Of if

				if (tempIndices[j] < tempMinimalIndex) {
					tempMinimalIndex = tempIndices[j];
					tempWeight = tempWeights[j];
					tempIndexInInicesArray = j;
				} // Of if
			} // of for j

			Triple tempNewTriple = new Triple();
			tempNewTriple.column = tempMinimalIndex;
			tempNewTriple.weight = tempWeight;
			tempProcessed[tempIndexInInicesArray] = true;

			// Link it now!
			tempReference.next = tempNewTriple;
			tempReference = tempNewTriple;
		} // Of for i

		// System.out.println("The vector is: " + resultHeader);

		return resultHeader;
	}// Of kNearestNeighbors

	/**
	 *********************
	 * Make the matrix symmetric.
	 *********************
	 */
	void symmetrize() {
		Triple tempTriple;
		int tempTarget;
		int tempI;
		double tempWeight;

		for (int i = 0; i < matrix.length; i++) {
			tempTriple = matrix[i].next;
			while (tempTriple != null) {
				tempTarget = tempTriple.column;
				tempI = i;
				tempWeight = tempTriple.weight;
				insertToList(tempTarget, tempI, tempWeight);

				tempTriple = tempTriple.next;
			} // Of while
		} // Of for i
	}// Of symmetrize

	/**
	 *********************
	 * Insert a triple to the list.
	 * @param paraTargetIndex The index of the list that will be inserted.
	 * @param paraI The inserted value.
	 * @param paraWeight The weight of the new triple.
	 *********************
	 */
	boolean insertToList(int paraTargetIndex, int paraI, double paraWeight) {
		boolean tempAlreadyExists = false;

		Triple tempP = matrix[paraTargetIndex];
		// Search the position
		Triple tempQ = tempP.next;

		while (tempQ != null) {
			if (tempQ.column == paraI) {
				tempAlreadyExists = true;
				break;
			} // Of if

			if (tempQ.column < paraI) {
				tempP = tempQ;
				tempQ = tempQ.next;
			} else {
				// Now insert
				totalInserted++;
				System.out.print("insert " + paraI + ", " + paraTargetIndex);
				Triple tempNewTriple = new Triple(paraI, paraWeight, tempQ);
				tempP.next = tempNewTriple;
				break;
			} // Of if
		} // Of while

		return !tempAlreadyExists;
	}// Of insertToList

	/**
	 *********************
	 * Make the matrix symmetric.
	 * @param paraMatrix The given matrix.
	 * @return A symmetric matrix.
	 *********************
	 */
	public static CompressedSymmetricMatrix weightMatrixToTransitionProbabilityMatrix(
			CompressedSymmetricMatrix paraMatrix) {
		CompressedSymmetricMatrix resultMatrix = new CompressedSymmetricMatrix(paraMatrix.matrix.length);

		double tempTotalWeight;
		Triple tempTriple, tempNewTriple, tempPreviousTriple;
		for (int i = 0; i < paraMatrix.matrix.length; i++) {
			// Scan this row for total weight
			tempTotalWeight = 0;
			tempTriple = paraMatrix.matrix[i].next;

			while (tempTriple != null) {
				tempTotalWeight += tempTriple.weight;
				tempTriple = tempTriple.next;
			} // Of while

			// Construct nodes and link
			tempTriple = paraMatrix.matrix[i].next;
			tempPreviousTriple = resultMatrix.matrix[i];
			while (tempTriple != null) {
				tempNewTriple = new Triple();
				tempNewTriple.column = tempTriple.column;
				tempNewTriple.weight = tempTriple.weight / tempTotalWeight;
				tempPreviousTriple.next = tempNewTriple;

				// One step further
				tempPreviousTriple = tempNewTriple;
				tempTriple = tempTriple.next;
			} // Of while

		} // Of for i
		return resultMatrix;
	}// Of weightMatrixToTransitionProbabilityMatrix

	/**
	 *********************
	 * For output.
	 *********************
	 */
	public String toString() {
		String resultString = "";
		Triple tempTriple;
		for (int i = 0; i < matrix.length; i++) {
			resultString += "\r\n" + i + ":";
			tempTriple = matrix[i].next;
			while (tempTriple != null) {
				resultString += "(" + tempTriple.column + "," + tempTriple.weight + "); ";
				tempTriple = tempTriple.next;
			} // Of while
		} // Of for i

		return resultString;
	}// Of toString

	/**
	 *********************
	 * Multiply matrices.
	 * @param paraMatrix1 The first matrix.
	 * @param paraMatrix2 The second matrix.
	 * @return The result.
	 *********************
	 */
	public static CompressedSymmetricMatrix multiply(CompressedSymmetricMatrix paraMatrix1,
			CompressedSymmetricMatrix paraMatrix2) {
		System.out.println("multiply test 1");
		CompressedSymmetricMatrix resultMatrix = new CompressedSymmetricMatrix(paraMatrix1.matrix.length);

		OrderedIntArray tempArray = new OrderedIntArray(10000);
		int tempMiddle;
		Triple tempOuterTriple, tempInnerTriple;
		// Step 1. Compute each row of the new matrix
		System.out.println("multiply test 2");
		for (int i = 0; i < paraMatrix1.matrix.length; i++) {
			tempArray.reset();
			// Step 1.1 Compute the array of available nodes
			tempOuterTriple = paraMatrix1.matrix[i].next;
			// System.out.println("multiply test 2.1");
			while (tempOuterTriple != null) {
				tempMiddle = tempOuterTriple.column;
				tempInnerTriple = paraMatrix2.matrix[tempMiddle].next;
				while (tempInnerTriple != null) {
					// System.out.print("\t" + i + "->" + tempMiddle + "->" +
					// tempInnerTriple.column);
					tempArray.insert(tempInnerTriple.column);
					tempInnerTriple = tempInnerTriple.next;
				} // Of inner while
				tempOuterTriple = tempOuterTriple.next;
			} // Of while
				// System.out.println("multiply test 2.2");
				// System.out.println(tempArray);

			// Step 1.2 Compute weights
			Triple tempPointer = resultMatrix.matrix[i];
			for (int j = 0; j < tempArray.size; j++) {
				// Add one node
				Triple tempNewTriple = new Triple();
				tempNewTriple.column = tempArray.data[j];
				tempNewTriple.weight = Triple.multiply(paraMatrix1.matrix[i], paraMatrix2.matrix[tempNewTriple.column]);
				tempPointer.next = tempNewTriple;
				tempPointer = tempNewTriple;
			} // Of for j
		} // Of for i

		return resultMatrix;
	}// Of multiply

}// Of CompressedSymmetricMatrix
