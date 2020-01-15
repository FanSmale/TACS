package clustering;

import java.util.Arrays;

import common.Common;
import common.DistanceMeasure;
import common.SimpleTools;
import exception.UnableToClusterInKException;
import weka.core.Instances;

/**
 * The super class of any density-based clustering algorithms. It help setting
 * the radius dc.
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

public abstract class DensityClustering extends Clustering {
	/**
	 * The distance threshold for density computation. For the density peak and
	 * DBScan algorithms.
	 */
	double dc;

	/**
	 * The diameter of the current block.
	 */
	double diameter;

	/**
	 * The average distance of the current block.
	 */
	double averageDistance;

	/**
	 * Compute the dc using the diameter.
	 */
	public static final int DIAMETER_FOR_DC = 0;

	/**
	 * Compute the dc using the average distance.
	 */
	public static final int AVERAGE_FOR_DC = 1;

	/**
	 * The scheme for computing dc. DIAMETER_FOR_DC or AVERAGE_FOR_DC.
	 */
	int dcScheme;

	/**
	 * The dc ratio.
	 */
	double dcRatio;

	/**
	 * The number of pairs for statistics. int numPairs;
	 */

	/**
	 * Default adaptive dc ratio.
	 */
	public static final double DEFAULT_DC_RATIO = 0.1;

	/**
	 * To speed up the computation of density.
	 */
	public static final int SPEED_UP_THRESHOLD = 1000;

	/**
	 * The max distance which cannot be exceeded..
	 */
	public static final double MAXIMAL_DISTANCE = 1e10;

	/**
	 * The density of each instance. For the density peak algorithm.
	 */
	double[] densities;

	/**
	 * Use cutoff kenrel to compute the density.
	 */
	public static final int CUTOFF_KERNEL = 0;

	/**
	 * Use Gaussian kernel to compute the density.
	 */
	public static final int GAUSSIAN_KERNEL = 1;

	/**
	 * The kernel.
	 */
	int kernel = 1;

	/**
	 * Smaller blocks for speed up. It is produced by the kMeans algorithm now.
	 */
	int[][] smallerBlocks;

	/**
	 ********************
	 * The constructor.
	 * 
	 * @param paraFilename
	 *            The data set filename.
	 * @param paraDistanceMeasure
	 *            The distance measure in integer.
	 * @param paraDcScheme
	 *            The dc scheme. DIAMETER_FOR_DC or AVERAGE_FOR_DC.
	 * @param paraDcRatio
	 *            The ratio for radius computation.
	 * @param paraKernel
	 *            The kernel, cutoff or Gaussian.
	 ********************
	 */
	public DensityClustering(String paraFilename, int paraDistanceMeasure, int paraDcScheme, double paraDcRatio,
			int paraKernel) {
		super(paraFilename, paraDistanceMeasure);

		dcScheme = paraDcScheme;
		dcRatio = paraDcRatio;
		kernel = paraKernel;

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
	public DensityClustering(Instances paraData, DistanceMeasure paraDistanceMeasure, int paraDcScheme,
			double paraDcRatio, int paraKernel) {
		super(paraData, paraDistanceMeasure);

		dcScheme = paraDcScheme;
		dcRatio = paraDcRatio;
		kernel = paraKernel;

		initialize();
	}// Of the second constructor

	/**
	 ********************
	 * Initialize.
	 ********************
	 */
	private void initialize() {
		setAdaptiveDc(dcRatio);
		densities = new double[numInstances];
		Arrays.fill(densities, 0);

		smallerBlocks = null;
		computeDensityEfficiently();
		SimpleTools.variableTrackingOutput("The densities are: " + densities[0] + "...\r\n");
	}// Of initialize

	/**
	 ************************* 
	 * Set dc adaptively according to the dataset.
	 * 
	 * @param paraRatio
	 *            The ratio of the average distance.
	 ************************* 
	 */
	public void setAdaptiveDc(double paraRatio) {
		setAdaptiveDc(wholeBlock, paraRatio);
	}// Of setAdaptiveDc

	/**
	 ************************* 
	 * Set dc adaptively according to the dataset.
	 * 
	 * @param paraBlock
	 *            The given block.
	 * @param paraRatio
	 *            The ratio of the average distance.
	 ************************* 
	 */
	public void setAdaptiveDc(int[] paraBlock, double paraRatio) {
		dcRatio = paraRatio;
		double tempTotal = 0;
		int tempFirst, tempSecond;
		int tempLength = paraBlock.length;
		int tempPairs = TIMES_FOR_FARTHEST_PAIR * tempLength;
		diameter = -1;
		double tempDistance;

		for (int i = 0; i < tempPairs; i++) {
			tempFirst = (int) (Common.random.nextDouble() * tempLength);
			tempSecond = (int) (Common.random.nextDouble() * tempLength);

			tempDistance = distanceMeasure.distance(paraBlock[tempFirst], paraBlock[tempSecond]);

			tempTotal += tempDistance;
			if (diameter < tempDistance) {
				diameter = tempDistance;
			} // Of if

			if (tempTotal > MAXIMAL_DISTANCE) {
				System.out.println("Error occurred in DensityClustering.setAdaptiveDc()");
				System.out.println("The distance between " + data.instance(tempFirst) + " and "
						+ data.instance(tempSecond) + " is infinity" + "(" + tempTotal + ")");
				System.exit(0);
			} // Of if
		} // Of for i

		averageDistance = tempTotal / tempPairs;

		dc = -1;
		if (dcScheme == DIAMETER_FOR_DC) {
			dc = diameter * dcRatio;
		} else if (dcScheme == AVERAGE_FOR_DC) {
			dc = averageDistance * dcRatio;
		} else {
			System.out.println("Fatal error in Clustering.setAdaptiveDc(int[], int, double):"
					+ "\r\nUnsupported dcScheme: " + dcScheme);
			System.exit(0);
		} // Of if

		SimpleTools.consoleOutput("tempTotal = " + tempTotal + ", tempPairs = " + tempPairs + ", paraRatio = "
				+ paraRatio + ", dc = " + dc);
	}// Of setAdaptiveDc

	/**
	 ************************* 
	 * Set dc.
	 * 
	 * @param paraDc
	 *            The given dc.
	 ************************* 
	 */
	public void setDc(double paraDc) {
		dc = paraDc;
	}// Of setDc

	/**
	 ********************
	 * Get the density of the specified instance.
	 * 
	 * @param paraIndex
	 *            The index of the instance.
	 * @return The density.
	 ********************
	 */
	public double getDensity(int paraIndex) {
		return densities[paraIndex];
	}// Of getDensity

	/**
	 ************************* 
	 * Test the method.
	 ************************* 
	 */
	public void testComputeDensity() {
		computeDensityEfficiently();
		SimpleTools.consoleOutput("dc = " + dc);
		SimpleTools.consoleOutput("The density array is: " + Arrays.toString(densities));
	}// Of testComputeDensity

	/**
	 ****************** 
	 * Compute the densities of instances in the given block.
	 * @param paraBlock The given block.
	 ****************** 
	 */
	public void computeDensities(int[] paraBlock) {
		switch (kernel) {
		case CUTOFF_KERNEL:
			SimpleTools.processTrackingOutput("CUTOFF_KERNEL ");
			computeDensitiesCutoff(paraBlock);
			break;
		case GAUSSIAN_KERNEL:
			SimpleTools.processTrackingOutput("GAUSSIAN_KERNEL ");
			computeDensitiesGaussian(paraBlock);
			break;
		default:
			System.out.println("Unsupported kernel: " + kernel);
			System.exit(0);
		}// Of switch
	}// Of computeDensities

	/**
	 ****************** 
	 * Compute the densities using cutoff.
	 ****************** 
	 */
	public void computeDensitiesCutoff() {
		computeDensitiesCutoff(wholeBlock);
	}// Of computeDensitiesCutoff

	/**
	 ****************** 
	 * Compute the densities using cutoff.
	 * 
	 * @param paraBlock
	 *            The given block for density computation. The density of each
	 *            instance in the block is computed, and only other instances in
	 *            the same block is considered.
	 ****************** 
	 */
	public void computeDensitiesCutoff(int[] paraBlock) {
		// SimpleTools.processTrackingOutput(
		// "Compute densities using cutoff for a block with " + paraBlock.length
		// + " instances.\r\n");

		double tempDistance;
		// Compute the densities.
		for (int i = 0; i < paraBlock.length; i++) {
			for (int j = 0; j < paraBlock.length; j++) {
				tempDistance = distanceMeasure.distance(paraBlock[i], paraBlock[j]);
				if (tempDistance <= dc) {
					densities[paraBlock[i]]++;
				} // Of if
			} // Of for j
		} // Of for i

		SimpleTools.variableTrackingOutput("The densities are " + Arrays.toString(densities));
	}// Of computeDensitiesCutoff

	/**
	 ****************** 
	 * Compute the densities using Gaussian kernel.
	 * @param paraBlock The given block.
	 ****************** 
	 */
	public void computeDensitiesGaussian(int[] paraBlock) {
		// SimpleTools
		// .processTrackingOutput("computeDensitiesGaussian for a block with " +
		// paraBlock.length + " instances.\r\n");

		double tempDistance;

		// Compute the densities.
		for (int i = 0; i < paraBlock.length; i++) {
			densities[paraBlock[i]] = 0;
			for (int j = 0; j < paraBlock.length; j++) {
				tempDistance = distanceMeasure.distance(paraBlock[i], paraBlock[j]);
				densities[paraBlock[i]] += Math.exp(-tempDistance * tempDistance / dc / dc);
			} // Of for j
		} // Of for i

		SimpleTools.variableTrackingOutput("The densities are " + Arrays.toString(densities) + "\r\n");
	}// Of computeDensitiesGaussian

	/**
	 ****************** 
	 * Compute the densities.
	 * 
	 * @param paraK
	 *            The number of blocks.
	 ****************** 
	 */
	public void computeSmallerBlocks(int paraK) {
		KMeans tempKMeans = new KMeans(data, distanceMeasure);
		// Try at most 5 times.
		boolean tempSuccess = false;
		for (int i = 0; i < 5; i++) {
			try {
				smallerBlocks = tempKMeans.clusterInK(paraK);
				tempSuccess = true;
			} catch (UnableToClusterInKException ee) {
				System.out.println("Error occurred in DensityClustering.computeDensityEfficiently().\r\n" + ee);
			} // Of try

			if (tempSuccess) {
				break;
			} // Of if
		} // Of for

		if (!tempSuccess) {
			System.out.println("Failed after trying 5 times.");
			System.exit(0);
		} // Of if
	}// Of computeSmallerBlocks

	/**
	 ************************* 
	 * Compute the density of each instance efficiently. It is implemented by
	 * first clustering using kMeans, where k = sqrt{n}
	 ************************* 
	 */
	public void computeDensityEfficiently() {
		Arrays.fill(densities, 0);
		// Unnecessary to speed up.
		if (numInstances <= SPEED_UP_THRESHOLD) {
			SimpleTools.processTrackingOutput("" + numInstances + " instances, small data, no need to speed up.\r\n");
			computeDensities(wholeBlock);
			return;
		} // Of if

		int tempK = 2;
		if (numInstances <= SPEED_UP_THRESHOLD * 100) {
			// Big.
			tempK = (int) (numInstances / SPEED_UP_THRESHOLD) + 1;
		} else {
			// Very big.
			tempK = 100; // (int) Math.sqrt(numInstances);
		} // Of if
		SimpleTools.processTrackingOutput("Bigger data, speed up with " + tempK + " blocks in DensityClustering.\r\n");

		// int tempK = (int)Math.sqrt(numInstances);

		computeSmallerBlocks(tempK);
		for (int i = 0; i < tempK; i++) {
			computeDensities(smallerBlocks[i]);
			SimpleTools.processTrackingOutput("" + i + ": " + smallerBlocks[i].length + " instances.\r\n");
		} // Of for i

		SimpleTools.processTrackingOutput("DensityClustering.computeDensityEfficiently() finished.\r\n");
		SimpleTools.processTrackingOutput(
				"densities = [" + densities[0] + ", ..., " + densities[numInstances - 1] + "]\r\n");
	}// Of computeDensityEfficiently
}// Of DensityClustering
