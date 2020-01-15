package clustering;

import java.util.Arrays;

import common.DistanceMeasure;
import common.SimpleTools;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import exception.UnableToClusterInKException;

public class WekaHierarchical extends Clustering {
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
	public WekaHierarchical(String paraFilename, int paraDistanceMeasure) {
		super(paraFilename, paraDistanceMeasure);
	}// Of the first constructor

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
	public WekaHierarchical(Instances paraData,
			DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure);
	}// Of the second constructor

	/**
	 ************************* 
	 * Cluster the given block in using kMeans.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraCenters
	 *            The given centers.
	 * @return Clusters
	 ************************* 
	 */
	public int[][] clusterInK(int[] paraBlock, int paraK)
			throws UnableToClusterInKException {
		// Step 1. Build the new data.
		Instances tempInstances = constructSubset(paraBlock);
		tempInstances.setClassIndex(-1);
		tempInstances.deleteAttributeAt(numConditions);
		// System.out.println("Small block constructed.");

		// Step 2. Cluster.
		HierarchicalClusterer tempHierarchical = new HierarchicalClusterer();
		DistanceFunction tempDistanceFunction = tempHierarchical.getDistanceFunction();
		System.out.println("tempDistanceFunction is " + tempDistanceFunction);
		tempHierarchical.setNumClusters(2);

		int[] tempAssignments = new int[paraBlock.length];
		Instance tempInstance;
		try {
			tempHierarchical.buildClusterer(tempInstances);
			// System.out.println("Hierarchical built.");

			for (int i = 0; i < paraBlock.length; i++) {
				tempInstance = tempInstances.instance(i);
				try {
					tempAssignments[i] = tempHierarchical
							.clusterInstance(tempInstance);
				} catch (Exception ee) {
					// Use default.
					tempAssignments[i] = 0;
				}// Of try
			}// Of for i

			// System.out.println("Assignment determined: "
			// + Arrays.toString(tempAssignments));
			clusters = blockInformationToBlocks(paraBlock, tempAssignments,
					paraK);
		} catch (Exception ee) {
			throw new UnableToClusterInKException(ee.toString());
		}// Of try

		return clusters;
	}// Of clusterInK

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
		SimpleTools.consoleOutput("Hello, kMeans.");
		String tempFilename = "src/data/iris.arff";
		// String tempFilename = "src/data/DLA.arff";

		if (args.length >= 1) {
			tempFilename = args[0];
			SimpleTools.consoleOutput("The filename is: " + tempFilename);
		} // Of if

		WekaHierarchical tempHierarchical = new WekaHierarchical(tempFilename,
				DistanceMeasure.EUCLIDEAN);

		// tempHierarchical.testClusterInTwo();
		tempHierarchical.testClusterInK(2);
	}// Of main

}// Of class WekaHierarchical
