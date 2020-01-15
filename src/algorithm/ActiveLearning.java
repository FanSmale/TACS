package algorithm;

import java.io.FileReader;
import java.util.Arrays;

import common.*;
import exception.*;
import weka.core.Instances;

/**
 * The superclass of any active learning algorithm. Read the data, compute the
 * accuracy, get misclassified ones.
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

public class ActiveLearning {

	/**
	 * The data.
	 */
	Instances data;

	/**
	 * The number of classes. For binary classification it is 2.
	 */
	int numClasses;

	/**
	 * The number of instances.
	 */
	int numInstances;

	/**
	 * The number of conditional attributes.
	 */
	int numConditions;

	/**
	 * Disorder the data for more trustworthy results.
	 */
	boolean disorder;

	/**
	 * The distance measure.
	 */
	DistanceMeasure distanceMeasure;

	/**
	 * The instance is unhandled yet.
	 */
	public static final int UNHANDLED = -1;

	/**
	 * The instance is queried.
	 */
	public static final int DEFAULT_LABELED = 0;

	/**
	 * The instance is queried.
	 */
	public static final int QUERIED = 1;

	/**
	 * The instance is predicted by pure block.
	 */
	public static final int PURE_BLOCK_PREDICTED = 2;

	/**
	 * The instance is predicted by kNN.
	 */
	public static final int KNN_PREDICTED = 3;

	/**
	 * Instance status array.
	 */
	int[] instanceStatusArray;

	/**
	 * The prediction, including queried and predicted labels.
	 */
	int[] predicts;

	/**
	 * The queried instance indices.
	 */
	private int[] querySequence;

	/**
	 * The actual number of queries.
	 */
	public int numQueries;

	/**
	 * Remaining queries provided by the oracle. The value is not initialized in
	 * this class.
	 */
	int numRemainingQueries;

	/**
	 * The default label. It is useful when we have no idea which label should
	 * be assigned.
	 */
	public static final int DEFAULT_LABEL = 0;

	/**
	 * The k value for kNN.
	 */
	int kValue;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraDistanceMeasure
	 *            The given distance measure in integer.
	 * @param paraNormalizeData
	 *            Normalize or not.
	 * @param paraDisorderData
	 *            Disorder data or not.
	 ********************
	 */
	public ActiveLearning(String paraFilename, int paraDistanceMeasure,
			boolean paraNormalizeData, boolean paraDisorderData) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n"
					+ ee);
			System.exit(0);
		} // Of try

		if (paraNormalizeData) {
			SimpleTools.normalizeDecisionSystem(data);
		} // Of if

		disorder = paraDisorderData;

		data.setClassIndex(data.numAttributes() - 1);
		numConditions = data.numAttributes() - 1;
		numInstances = data.numInstances();
		numClasses = data.attribute(numConditions).numValues();

		setDistanceMeasure(paraDistanceMeasure);
		// distanceMeasure = new DistanceMeasure(data, paraDistanceMeasure);
		querySequence = new int[numInstances];
		predicts = new int[numInstances];
		Arrays.fill(predicts, UNHANDLED);
		// isQueriedArray = new boolean[numInstances];

		numQueries = 0;
		instanceStatusArray = new int[numInstances];
		Arrays.fill(instanceStatusArray, UNHANDLED);
		
		kValue = 1;
	}// Of the constructor

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 * @param paraDistanceMeasure
	 *            The given distance measure in integer.
	 ********************
	 */
	public ActiveLearning(String paraFilename, int paraDistanceMeasure) {
		this(paraFilename, paraDistanceMeasure, true, false);
	}// Of the second constructor

	/**
	 ********************
	 * Reset for repeated running.
	 ********************
	 */
	public void reset() {
		numQueries = 0;
		Arrays.fill(predicts, UNHANDLED);
		Arrays.fill(instanceStatusArray, UNHANDLED);

		if (disorder) {
			SimpleTools
					.processTrackingOutput("ActiveLearning.reset(), Disorder data ...");
			SimpleTools.disorderData(data);
			SimpleTools.processTrackingOutput("done.\r\n");
		} // Of if

		setDistanceMeasure(distanceMeasure.getMeasure());
	}// Of reset

	/**
	 ************************* 
	 * Is the instance queried?
	 * 
	 * @param paraIndex
	 *            The index of the instance.
	 * @return Whether or not the instance is queried.
	 ************************* 
	 *         public boolean isQueried(int paraIndex) { if
	 *         (instanceStatusArray[paraIndex] == QUERIED) { return true; } //
	 *         Of if return false; }// of isQueried
	 */

	/**
	 ************************* 
	 * Set the k value for kNN.
	 * @param paraKValue The given k value.
	 ************************* 
	 */
	public void setKValue(int paraKValue) {
		kValue = paraKValue;
	}//Of setKValue
	
	/**
	 ************************* 
	 * Get the query sequence.
	 * 
	 * @return The query sequence ordered by the query time, e.g., [48, 7, 36].
	 ************************* 
	 */
	public int[] getQuerySequence() {
		int[] resultQuerySequence = new int[numQueries];

		for (int i = 0; i < numQueries; i++) {
			resultQuerySequence[i] = querySequence[i];
		} // Of for i

		return resultQuerySequence;
	}// Of getQuerySequence

	/**
	 ************************* 
	 * Set the distance measure.
	 * 
	 * @param paraDistanceMeasure
	 *            The distance measure in int.
	 ************************* 
	 */
	public void setDistanceMeasure(int paraDistanceMeasure) {
		distanceMeasure = new DistanceMeasure(data, paraDistanceMeasure);
	}// Of setDistanceMeasure

	/**
	 ************************* 
	 * Get the distance measure.
	 * 
	 * @return The distance measure in int.
	 ************************* 
	 */
	public int getDistanceMeasure() {
		return distanceMeasure.getMeasure();
	}// Of getDistanceMeasure

	/**
	 ************************* 
	 * How many instances are with the given status?
	 * 
	 * @param paraStatus
	 *            The status, UNHANDLED, DEFAULT_LABELED, etc.
	 * @return The number of instances with the given status.
	 ************************* 
	 */
	public int getNumInstancesByStatus(int paraStatus) {
		if ((paraStatus < UNHANDLED) || (paraStatus > KNN_PREDICTED)) {
			System.out
					.println("Internal error in ActiveLearning.getNumInstancesByStatus(int)\r\n"
							+ "The status " + paraStatus + " is illegal.");
			System.exit(0);
		} // Of if

		int resultCount = 0;
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] == paraStatus) {
				resultCount++;
			} // Of if
		} // Of for i

		return resultCount;
	}// Of getNumInstancesByStatus

	/**
	 ************************* 
	 * Which instances have the given status?
	 * 
	 * @param paraStatus
	 *            The given status.
	 * 
	 * @return The indices of instances corresponding to given status.
	 ************************* 
	 */
	public int[] getInstancesByStatus(int paraStatus) {
		int tempLength = getNumInstancesByStatus(paraStatus);

		int[] resultArray = new int[tempLength];
		int tempCount = 0;
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] == paraStatus) {
				resultArray[tempCount] = i;
				tempCount++;
			} // Of if
		} // Of for i

		return resultArray;
	}// Of getInstancesByStatus

	/**
	 ************************* 
	 * Get the number of queries.
	 * 
	 * @return The number of queries.
	 ************************* 
	 */
	public int getNumQueries() {
		return numQueries;
	}// Of getNumQueries

	/**
	 ************************* 
	 * set the query fraction.
	 * 
	 * @param paraFraction
	 *            The fraction.
	 ************************* 
	 */
	public void setQueryFraction(double paraFraction) {
		numRemainingQueries = (int) (numInstances * paraFraction);
	}// Of setQueryFraction

	/**
	 ************************* 
	 * Query a label under control.
	 * 
	 * @param paraIndex
	 *            The index of the queried instance.
	 * @throws LabelUsedUpException
	 *             If labels are used up.
	 * @throws DuplicateQueryException
	 *             If an instance is queried more than one time.
	 ************************* 
	 */
	public void query(int paraIndex) throws LabelUsedUpException,
			DuplicateQueryException {
		// No more queries are permitted.
		if (numRemainingQueries <= 0) {
			throw new LabelUsedUpException(
					"Queries are used up while trying to query instance #"
							+ paraIndex + ".");
		} // Of if

		// The instance is already handled.
		if (predicts[paraIndex] >= 0) {
			throw new DuplicateQueryException("Instance #" + paraIndex
					+ " has a label already.");
		} // Of if

		// Now execute these operations.
		predicts[paraIndex] = (int) data.instance(paraIndex).classValue();
		changeInstanceStatus(paraIndex, QUERIED);
		numRemainingQueries--;

		querySequence[numQueries] = paraIndex;
		numQueries++;

		// System.out.print(" q(" + paraIndex + ")");
	}// Of query

	/**
	 ************************* 
	 * Change the instance status.
	 * 
	 * @param paraIndex
	 *            The index of the instance.
	 * @param paraNewStatus
	 *            The new status of the instance.
	 ************************* 
	 */
	public void changeInstanceStatus(int paraIndex, int paraNewStatus) {
		// Cannot change any status to UNHANDLED.
		if (paraNewStatus == UNHANDLED) {
			System.out
					.println("Error occurred in ActiveLearning.changeInstanceStatus(int, int)\r\n"
							+ "Cannot change any status to UNHANDLED.");
			System.exit(0);
		} // Of if

		if (instanceStatusArray[paraIndex] != UNHANDLED) {
			System.out
					.println("Error occurred in ActiveLearning.changeInstanceStatus(int, int)\r\n"
							+ "Cannot change a status from "
							+ instanceStatusArray[paraIndex]
							+ " to any others such as " + paraNewStatus + ".");
			System.exit(0);
		} // Of if

		instanceStatusArray[paraIndex] = paraNewStatus;
	}// Of changeInstanceStatus

	/**
	 ************************* 
	 * Compute the accuracy (not including queried ones.)
	 * 
	 * @return The accuracy.
	 ************************* 
	 */
	public double computeAccuracy() {
		System.out.println(Arrays.toString(instanceStatusArray));
		double tempIncorrect = 0;
		for (int i = 0; i < predicts.length; i++) {
			if (instanceStatusArray[i] == QUERIED) {
				continue;
			} // Of if

			if (predicts[i] != (int) data.instance(i).classValue()) {
				tempIncorrect++;
			} // Of if
		} // Of for i

		int tempNumQueries = getNumQueries();

		double tempAccuracy = (data.numInstances() - tempNumQueries - tempIncorrect)
				/ (data.numInstances() - tempNumQueries);
		return tempAccuracy;
	}// Of computeAccuracy

	/**
	 ************************* 
	 * Get the number of misclassified instances.
	 * 
	 * @return The number of misclassified instances.
	 ************************* 
	 */
	public int getNumMisclassified() {
		int resultNumMisclassified = 0;

		// Step 1. Add to the array.
		for (int i = 0; i < predicts.length; i++) {
			if (predicts[i] != (int) data.instance(i).classValue()) {
				resultNumMisclassified++;
			} // Of if
		} // Of for i

		return resultNumMisclassified;
	}// Of getNumMisclassified

	/**
	 ************************* 
	 * Get the number of unhandled instances.
	 * 
	 * @return The number of unhandled instances.
	 ************************* 
	 */
	public int getNumUnhandled() {
		int resultNumUnhandled = 0;

		// Step 1. Add to the array.
		for (int i = 0; i < instanceStatusArray.length; i++) {
			if (instanceStatusArray[i] == UNHANDLED) {
				resultNumUnhandled++;
			} // Of if
		} // Of for i

		return resultNumUnhandled;
	}// Of getNumUnhandled	
	
	/**
	 ************************* 
	 * Get misclassified instances.
	 * 
	 * @return The indices of misclassified instances, such as [5, 16], or empty
	 *         array [].
	 ************************* 
	 */
	public int[] getMisclassified() {
		int tempNumMisclassify = getNumMisclassified();
		int[] resultMisclassifies = new int[tempNumMisclassify];

		// Step 1. Add to the array.
		int tempCounter = 0;
		for (int i = 0; i < predicts.length; i++) {
			if (predicts[i] != (int) data.instance(i).classValue()) {
				resultMisclassifies[tempCounter] = i;
				tempCounter++;
			} // Of if
		} // Of for i

		return resultMisclassifies;
	}// Of getMisclassified

	/**
	 ************************* 
	 * Get the number of misclassified instances under the given status.
	 * 
	 * @param paraStatus
	 *            DEFAULT_LABELED, PURE_BLOCK_PREDICTED, or ONE_NN_PREDICTED
	 * @return The number of misclassified instances.
	 ************************* 
	 */
	public int getNumMisclassified(int paraStatus) {
		int resultNumMisclassified = 0;

		// Step 1. Add to the array.
		for (int i = 0; i < predicts.length; i++) {
			if (instanceStatusArray[i] != paraStatus) {
				continue;
			} // of if

			if (predicts[i] != (int) data.instance(i).classValue()) {
				resultNumMisclassified++;
			} // Of if
		} // Of for i

		return resultNumMisclassified;
	}// Of getNumPureBlockMisclassified

	/**
	 ************************* 
	 * Get misclassified instances in "pure" blocks. That is because some blocks
	 * are not really pure.
	 * 
	 * @param paraStatus
	 *            The status for checking. It can be PURE_BLOCK_PREDICTED,
	 *            ONE_NN_PREDICTED, etc.
	 * @return The indices of instances, such as [5, 16], or empty array [].
	 ************************* 
	 */
	public int[] getMisclassified(int paraStatus) {
		int tempNumMisclassify = getNumMisclassified(paraStatus);
		int[] resultMisclassifies = new int[tempNumMisclassify];

		// Step 1. Add to the array.
		int tempCounter = 0;
		for (int i = 0; i < predicts.length; i++) {
			if (instanceStatusArray[i] != paraStatus) {
				continue;
			} // Of if

			if (predicts[i] != (int) data.instance(i).classValue()) {
				resultMisclassifies[tempCounter] = i;
				tempCounter++;
			} // Of if
		} // Of for i

		return resultMisclassifies;
	}// Of getMisclassified

	/**
	 ************************* 
	 * Classify unhandled instances using kNN.
	 ************************* 
	 */
	public void knnUnhandled() {
		knnUnhandled(kValue);
	}// Of knnUnhandled
	
	/**
	 ************************* 
	 * Classify unhandled instances using kNN.
	 * 
	 * @param paraK
	 *            The given k value.
	 ************************* 
	public void knnUnhandled(int paraK) {
		for (int i = 0; i < predicts.length; i++) {
			if (predicts[i] == UNHANDLED) {
				predicts[i] = knn(i, paraK);
				changeInstanceStatus(i, KNN_PREDICTED);
			} // Of if
		} // Of for i
	}// Of knnUnhandled
	 */
	
	/**
	 ************************* 
	 * Classify unhandled instances using 1NN. The neighbor should be in the
	 * same block. In case no label has been queried, the default label will be
	 * 0.
	 * 
	 * @param paraK
	 *            The k value.
	 ************************* 
	 */
	public void knnUnhandled(int paraK) {
		// Step 1. How many instances have been queried?
		int tempNumQueried = 0;
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] == QUERIED) {
				tempNumQueried++;
			} // Of if
		} // Of for i
		
		//Step 2. Should have enough queried labels.
		if (tempNumQueried < paraK) {
			System.out.println("Error occurred in ActiveLearning.knnUnhandled(int):\r\n"
					+ "No enough queried instances as neighbors.");
			System.exit(0);
		}//Of if

		// Step 3. Construct the queried array. This approach saves time.
		int[] tempQueriedArray = new int[tempNumQueried];
		int tempCounter = 0;
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] == QUERIED) {
				tempQueriedArray[tempCounter] = i;
				tempCounter++;
			} // Of if
		} // Of for i

		// Step 3. Find the nearest neighbor.
		double tempMinDistance;
		double tempDistance;
		int tempClosest;
		for (int i = 0; i < numInstances; i++) {
			if (instanceStatusArray[i] != UNHANDLED) {
				continue;
			} // Of if

			tempMinDistance = Double.MAX_VALUE;
			tempClosest = -1;

			for (int j = 0; j < tempNumQueried; j++) {
				tempDistance = distanceMeasure.distance(i,
						tempQueriedArray[j]);
				if (tempMinDistance > tempDistance) {
					tempMinDistance = tempDistance;
					tempClosest = tempQueriedArray[j];
				} // Of if
			} // Of for j

			predicts[i] = predicts[tempClosest];
			changeInstanceStatus(i, KNN_PREDICTED);
		} // Of for i
	}// Of knnUnhandled

	/**
	 ************************* 
	 * Classify an instance using kNN.
	 * 
	 * @param paraIndex
	 *            The given instance to classify
	 * @return The classification.
	 ************************* 
	 */
	private int knn(int paraIndex, int paraK) {
		int[] tempNearests = new int[paraK + 2];
		double[] tempDistances = new double[paraK + 2];

		// Step 1. Initialize
		Arrays.fill(tempNearests, -1);
		Arrays.fill(tempDistances, Double.MAX_VALUE);
		tempDistances[0] = -1;

		// Step 2. Find neighbors
		DistanceMeasure tempMeasure = new DistanceMeasure(data,
				DistanceMeasure.EUCLIDEAN);
		double tempCurrentDistance;
		for (int i = 0; i < numInstances; i++) {
			if (predicts[i] != QUERIED) {
				continue;
			} // Of if

			tempCurrentDistance = tempMeasure.distance(paraIndex, i);
			for (int j = paraK;; j--) {
				if (tempCurrentDistance < tempDistances[j]) {
					tempDistances[j + 1] = tempDistances[j];
					tempNearests[j + 1] = tempNearests[j];
				} else {
					tempDistances[j + 1] = tempCurrentDistance;
					tempNearests[j + 1] = i;
					break;
				} // Of if
			} // Of for j
		} // Of for i

		// Step 3. Vote and return
		int[] tempVotes = new int[numClasses];
		for (int i = 1; i <= paraK; i++) {
			tempVotes[predicts[tempNearests[i]]]++;
		} // Of for i
		SimpleTools.consoleOutput("The votes are: "
				+ Arrays.toString(tempVotes));

		int tempMajority = -1;
		int tempMaximalVotes = -1;
		for (int i = 0; i < tempVotes.length; i++) {
			if (tempVotes[i] > tempMaximalVotes) {
				tempMaximalVotes = tempVotes[i];
				tempMajority = i;
			} // Of if
		} // Of for i

		return tempMajority;
	}// Of knn

	/**
	 ************************* 
	 * Learn. This method should be overwritten in subclasses. The current code
	 * is for test only.
	 * 
	 * @return The result information.
	 ************************* 
	 */
	public String learn() {
		String resultString = "";
		// Query numRemainingQueries labels
		try {
			for (int i = 0; i < numInstances; i++) {
				int tempIndex = Common.random.nextInt(numInstances);
				if (instanceStatusArray[tempIndex] != QUERIED) {
					query(tempIndex);
				} // Of if
			} // Of for i
		} catch (Exception ee) {
			// Control the number of queries.
		} // Of try

		knnUnhandled(3);

		resultString += "\r\nI have queried " + getNumQueries()
				+ " labels randomly.\r\n";
		resultString += "With 3NN, the accuracy is: " + computeAccuracy()
				+ "\r\n";
		resultString += "The misclassified are: "
				+ Arrays.toString(getMisclassified()) + "\r\n";

		return resultString;
	}// Of learn

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
		System.out
				.println("Hello, active learning. I only want to test the constructor and kNN.");
		//String tempFilename = "src/data/mushroom.arff";
		// String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/r15.arff";
		// String tempFilename = "src/data/banana.arff";
		String tempFilename = "src/data/kr-vs-kp_nominal.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		ActiveLearning tempLearner = new ActiveLearning(tempFilename,
				DistanceMeasure.EUCLIDEAN, true, false);
		tempLearner.setQueryFraction(0.8);
		String tempResults = tempLearner.learn();

		System.out.println(tempResults);
	}// Of main

}// Of class ActiveLearning
