package clustering;

import java.util.Arrays;

import common.DistanceMeasure;
import common.SimpleTools;
import weka.clusterers.DBScan;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import exception.UnableToClusterInKException;

public class WekaDBScan extends DensityClustering {
	/**
	 * The EPS ratio.
	 */
	public static final double DEFAULT_EPS_RATIO = 0.1;

	/**
	 * The EPS ratio. double epsRatio;
	 */

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
	public WekaDBScan(String paraFilename, int paraDistanceMeasure,
			int paraKernel) {
		super(paraFilename, paraDistanceMeasure, AVERAGE_FOR_DC, 0.2,
				paraKernel);
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
	public WekaDBScan(String paraFilename, int paraDistanceMeasure) {
		this(paraFilename, paraDistanceMeasure, CUTOFF_KERNEL);
	}// Of the second constructor

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
	public WekaDBScan(Instances paraData, DistanceMeasure paraDistanceMeasure) {
		super(paraData, paraDistanceMeasure, AVERAGE_FOR_DC, 0.2, 0);
	}// Of the third constructor

	/**
	 ********************
	 * The constructor for independent running.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure as an object.
	 * @param paraKernel
	 *            The given kernel.
	 ********************
	 */
	public WekaDBScan(Instances paraData, DistanceMeasure paraDistanceMeasure,
			double paraAdaptiveRatio, int paraKernel) {
		super(paraData, paraDistanceMeasure, AVERAGE_FOR_DC, paraAdaptiveRatio,
				paraKernel);
	}// Of the third constructor

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
		DBScan tempDBSCan = new DBScan();
		double tempEpsilon;
		boolean tempSuccess = false;
		Exception tempEe = null;
		for (tempEpsilon = dc; tempEpsilon <= 0.501; tempEpsilon += 0.1) {
			tempDBSCan.setEpsilon(tempEpsilon);
			int[] tempAssignments = new int[paraBlock.length];
			Instance tempInstance;
			try {
				tempDBSCan.buildClusterer(tempInstances);
				// System.out.println("DBScan built.");

				for (int i = 0; i < paraBlock.length; i++) {
					tempInstance = tempInstances.instance(i);
					// System.out.print("Trying to cluster " + +paraBlock[i] +
					// ": "
					// + tempInstance);
					try {
						tempAssignments[i] = tempDBSCan
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

				// No exception
				System.out.println("tempEpsilon = " + tempEpsilon);
				tempSuccess = true;
				break;
			} catch (Exception ee) {
				tempEe = ee;
			}// Of try
		}// Of for tempEpsilon

		if (!tempSuccess) {
			throw new UnableToClusterInKException(tempEe.toString());
		}// Of if

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

		WekaDBScan tempDBScan = new WekaDBScan(tempFilename,
				DistanceMeasure.EUCLIDEAN);

		// tempDBScan.testClusterInTwo();
		tempDBScan.testClusterInK(2);
	}// Of main

}// Of class WekaDBScan
