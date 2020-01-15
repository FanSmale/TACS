package clustering;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import common.DistanceMeasure;
import common.SimpleTools;
import exception.UnableToClusterInKException;

public class WekaKMeans extends MeansClustering {
	
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
	public WekaKMeans(String paraFilename, int paraDistanceMeasure) {
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
	public WekaKMeans(Instances paraData, DistanceMeasure paraDistanceMeasure) {
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
	public int[][] clusterInK(int[] paraBlock, double[][] paraCenters) throws UnableToClusterInKException {
		// Number of blocks.
		int tempK = paraCenters.length;
		
		//Step 1. Build the new data.
		Instances tempIntances = constructSubset(paraBlock);
		tempIntances.setClassIndex(-1);
		tempIntances.deleteAttributeAt(numConditions);
		
		//Step 2. Cluster.
		SimpleKMeans tempKMeans = new SimpleKMeans();
		int[] tempAssignments = null;
		try {
			tempKMeans.setPreserveInstancesOrder(true);
			tempKMeans.setNumClusters(tempK);
			tempKMeans.buildClusterer(tempIntances);
			tempAssignments = tempKMeans.getAssignments();
			clusters = blockInformationToBlocks(paraBlock, tempAssignments, tempK);
		} catch (Exception ee) {
			throw new UnableToClusterInKException(ee.toString());
		}//Of try
		
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

		WekaKMeans tempkMeans = new WekaKMeans(tempFilename, DistanceMeasure.EUCLIDEAN);

		tempkMeans.testClusterInTwo();
		//tempkMeans.testClusterInK(3);
	}// Of main
}//Of class WekaKMeans 
