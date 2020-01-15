package gui;

import java.awt.*;
import java.awt.event.*;
import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.Date;

import javax.swing.JComboBox;

import algorithm.*;
import common.Common;
import common.SimpleTools;
import gui.guicommon.*;
import gui.guidialog.common.ErrorDialog;
import gui.guidialog.common.HelpDialog;
import gui.others.*;

/**
 * Three-way active learning through clustering algorithm selection.
 * <p>
 * See Min F. et al. Three-way active learning through clustering algorithm selection,
 * International Journal of Machine Learning and Cybernetics, 2020.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Almost finished, further revision is possible.<br>
 * Written time: May 20, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public class Tacs implements ActionListener, ItemListener {
	/**
	 * Select the arff file.
	 */
	private FilenameField arffFilenameField;

	/**
	 * Indicate number of queries?
	 */
	private Checkbox indicateQueriesCheckbox;

	/**
	 * The proportion of labels that can be queried.
	 */
	private DoubleField queryFractionField;

	/**
	 * The proportion of representative labels queried in the first round.
	 */
	private DoubleField representativeFractionField;

	/**
	 * Checkboxes for algorithms.
	 */
	private Checkbox[] algorithmCheckboxes;

	/**
	 * Distance measures: Euclidean, Manhattan, Mahalanobis
	 */
	private JComboBox<String> distanceJComboBox;

	/**
	 * Ensemble algorithm
	 */
	private JComboBox<String> ensembleAlgorithmJComboBox;

	/**
	 * Critical instance selection strategy
	 */
	private JComboBox<String> selectCriticalStrategyJComboBox;

	/**
	 * Query amount for critical instance selection
	 */
	private JComboBox<String> queryAmountStrategyJComboBox;

	/**
	 * Retrospect or not.
	 */
	private Checkbox retrospectCheckbox;

	/**
	 * Normalize or not.
	 */
	private Checkbox normalizeCheckbox;

	/**
	 * Disorder or not.
	 */
	private Checkbox disorderCheckbox;

	/**
	 * For density computation of Density Peaks (maybe also others.)
	 */
	private DoubleField adaptiveRatioDoubleField;

	/**
	 * Small block threshold. Small blocks will not be classified using the pure
	 * criteria.
	 */
	private IntegerField smallBlockThresholdIntegerField;

	/**
	 * The k value for kNN.
	 */
	private IntegerField kValueIntegerField;

	/**
	 * For neighbor based weight as well as entropy computation.
	 */
	private DoubleField neighborBasedWeightDoubleField;

	/**
	 * Checkbox for variable tracking.
	 */
	private Checkbox variableTrackingCheckbox;

	/**
	 * Checkbox for variable tracking.
	 */
	private Checkbox processTrackingCheckbox;

	/**
	 * Result output to file checkbox.
	 */
	private Checkbox fileOutputCheckbox;

	/**
	 * The message area.
	 */
	private TextArea messageTextArea;

	/**
	 * How many times to repeat.
	 */
	private IntegerField repeatTimesField;

	/**
	 *************************** 
	 * The only constructor.
	 *************************** 
	 */
	public Tacs() {
		// A simple frame to contain dialogs.
		Frame mainFrame = new Frame();
		mainFrame
				.setTitle("Three-way active learning through clustering selection. github.com/FanSmale/Tacs, minfanphd@163.com");
		// The top part: select arff file.
		arffFilenameField = new FilenameField(30);
		arffFilenameField.setText("src/data/iris.arff");
		Button browseButton = new Button(" Browse ");
		browseButton.addActionListener(arffFilenameField);

		Panel sourceFilePanel = new Panel();
		sourceFilePanel.add(new Label("The .arff file:"));
		sourceFilePanel.add(arffFilenameField);
		sourceFilePanel.add(browseButton);

		// How many labels can be queries, and how many can be queried in the
		// first round
		Panel numQueriesPanel = new Panel();
		numQueriesPanel.setLayout(new FlowLayout());
		indicateQueriesCheckbox = new Checkbox("Indicate queries", true);
		queryFractionField = new DoubleField("0.2", 5);
		representativeFractionField = new DoubleField("0.5", 5);
		numQueriesPanel.add(indicateQueriesCheckbox);
		numQueriesPanel.add(new Label("Query fraction: "));
		numQueriesPanel.add(queryFractionField);
		numQueriesPanel.add(new Label("Representative fraction: "));
		numQueriesPanel.add(representativeFractionField);
		queryFractionField.setEnabled(true);
		representativeFractionField.setEnabled(true);
		indicateQueriesCheckbox.addItemListener(this);

		Label distanceLabel = new Label("Distance measure: ");
		String[] distances = { "Euclidean", "Manhattan", "Mahalanobis" };
		distanceJComboBox = new JComboBox<String>(distances);
		distanceJComboBox.setSelectedIndex(0);
		retrospectCheckbox = new Checkbox(" Retrospect ", false);
		normalizeCheckbox = new Checkbox(" Normalize ", true);
		disorderCheckbox = new Checkbox(" Disorder ", true);

		Label tempEnsembleAlgorithmsLabel = new Label("Ensemble algorithm");
		String[] tempEnsembleAlgorithms = { "Selector", "Majority union" };
		ensembleAlgorithmJComboBox = new JComboBox<String>(
				tempEnsembleAlgorithms);
		ensembleAlgorithmJComboBox.setSelectedIndex(0);

		Panel distancePanel = new Panel();
		distancePanel.add(distanceLabel);
		distancePanel.add(distanceJComboBox);
		distancePanel.add(tempEnsembleAlgorithmsLabel);
		distancePanel.add(ensembleAlgorithmJComboBox);
		distancePanel.add(retrospectCheckbox);

		Panel tempPreprocessingPanel = new Panel();
		tempPreprocessingPanel.add(normalizeCheckbox);
		tempPreprocessingPanel.add(disorderCheckbox);

		String[] tempSelectCriticalStrategyAlgorithms = { "DP representative",
				"Max total distance" };
		selectCriticalStrategyJComboBox = new JComboBox<String>(
				tempSelectCriticalStrategyAlgorithms);
		selectCriticalStrategyJComboBox.setSelectedIndex(0);
		String[] tempQueryAmountStrategyAlgorithms = { "Sqrt(n) at once",
				"Sqrt(n) or impure" };
		queryAmountStrategyJComboBox = new JComboBox<String>(
				tempQueryAmountStrategyAlgorithms);
		queryAmountStrategyJComboBox.setSelectedIndex(0);

		Panel tempInstanceSelectionPanel = new Panel();
		tempInstanceSelectionPanel.add(new Label(" Select critical: "));
		tempInstanceSelectionPanel.add(selectCriticalStrategyJComboBox);
		tempInstanceSelectionPanel.add(new Label(" Query amount: "));
		tempInstanceSelectionPanel.add(queryAmountStrategyJComboBox);

		neighborBasedWeightDoubleField = new DoubleField("0.7");
		Panel neighborBasedWeightPanel = new Panel();
		neighborBasedWeightPanel.add(new Label(" Weight for entropy: "));
		neighborBasedWeightPanel.add(neighborBasedWeightDoubleField);

		adaptiveRatioDoubleField = new DoubleField("0.2");
		// Panel adaptiveRatioPanel = new Panel();
		neighborBasedWeightPanel
				.add(new Label(" Adaptive ratio for density: "));
		neighborBasedWeightPanel.add(adaptiveRatioDoubleField);

		Panel thresholdPanel = new Panel();
		smallBlockThresholdIntegerField = new IntegerField("10");
		thresholdPanel.add(new Label(" Small block threshold: "));
		thresholdPanel.add(smallBlockThresholdIntegerField);
		kValueIntegerField = new IntegerField("3");
		thresholdPanel.add(new Label(" k (for kNN): "));
		thresholdPanel.add(kValueIntegerField);

		processTrackingCheckbox = new Checkbox(" Process tracking ", false);
		variableTrackingCheckbox = new Checkbox(" Variable tracking ", false);
		fileOutputCheckbox = new Checkbox(" Output to file ", false);
		Panel trackingPanel = new Panel();
		trackingPanel.add(processTrackingCheckbox);
		trackingPanel.add(variableTrackingCheckbox);
		trackingPanel.add(fileOutputCheckbox);

		Panel topPanel = new Panel();
		topPanel.setLayout(new GridLayout(8, 1));
		topPanel.add(sourceFilePanel);
		topPanel.add(numQueriesPanel);
		topPanel.add(distancePanel);
		topPanel.add(tempPreprocessingPanel);
		topPanel.add(tempInstanceSelectionPanel);
		topPanel.add(neighborBasedWeightPanel);
		topPanel.add(thresholdPanel);
		topPanel.add(trackingPanel);

		// The middle part: algorithms
		algorithmCheckboxes = new Checkbox[ClusteringAlgorithmSelectionActiveLearning.NUM_ALGORITHMS];
		algorithmCheckboxes[0] = new Checkbox(" DP-Gaussian ", true);
		algorithmCheckboxes[1] = new Checkbox(" kMeans ", true);
		algorithmCheckboxes[2] = new Checkbox(" Hierarchical ", true);
		algorithmCheckboxes[3] = new Checkbox(" DBScan ", false);
		algorithmCheckboxes[4] = new Checkbox(" FCM ", false);
		algorithmCheckboxes[5] = new Checkbox(" Random walks ", false);
		algorithmCheckboxes[6] = new Checkbox(" DP-cutoff ", false);

		Panel algorithmsPanel = new Panel();
		algorithmsPanel.setLayout(new GridLayout(4, 3));
		algorithmsPanel.add(new Label("------------------------------------"));
		algorithmsPanel.add(new Label("Candidate clustering algorithms"));
		algorithmsPanel.add(new Label("------------------------------------"));
		for (int i = 0; i < ClusteringAlgorithmSelectionActiveLearning.NUM_ALGORITHMS; i++) {
			algorithmsPanel.add(algorithmCheckboxes[i]);
		} // Of for i

		Panel centralPanel = new Panel();
		centralPanel.setLayout(new GridLayout(2, 1));
		centralPanel.add(algorithmsPanel);
		messageTextArea = new TextArea(80, 40);
		centralPanel.add(messageTextArea);

		// The bottom part: ok and exit
		repeatTimesField = new IntegerField("1");
		Panel repeatTimesPanel = new Panel();
		repeatTimesPanel.add(new Label(" Repeat times: "));
		repeatTimesPanel.add(repeatTimesField);

		Button okButton = new Button(" OK ");
		okButton.addActionListener(this);
		// DialogCloser dialogCloser = new DialogCloser(this);
		Button exitButton = new Button(" Exit ");
		// cancelButton.addActionListener(dialogCloser);
		exitButton.addActionListener(ApplicationShutdown.applicationShutdown);
		Button helpButton = new Button(" Help ");
		helpButton.setSize(20, 10);
		HelpDialog helpDialog = null;
		try {
			helpDialog = new HelpDialog("Alcs algorithm",
					"src/gui/tacsHelp.txt");
			helpButton.addActionListener(helpDialog);
		} catch (Exception ee) {
			try {
				helpDialog = new HelpDialog("Alcs algorithm",
						"gui/tacsHelp.txt");
				helpButton.addActionListener(helpDialog);
			} catch (Exception ee2) {
				ErrorDialog.errorDialog.setMessageAndShow(ee.toString());
			}// Of try
		}// Of try
		Panel okPanel = new Panel();
		okPanel.add(okButton);
		okPanel.add(exitButton);
		okPanel.add(helpButton);

		Panel southPanel = new Panel();
		southPanel.setLayout(new GridLayout(2, 1));
		southPanel.add(repeatTimesPanel);
		southPanel.add(okPanel);

		mainFrame.setLayout(new BorderLayout());
		mainFrame.add(BorderLayout.NORTH, topPanel);
		mainFrame.add(BorderLayout.CENTER, centralPanel);
		mainFrame.add(BorderLayout.SOUTH, southPanel);

		mainFrame.setSize(700, 700);
		mainFrame.setLocation(10, 10);
		mainFrame.addWindowListener(ApplicationShutdown.applicationShutdown);
		mainFrame.setBackground(GUICommon.MY_COLOR);
		mainFrame.setVisible(true);
	}// Of the constructor

	/**
	 *************************** 
	 * Read the arff file.
	 *************************** 
	 */
	public void actionPerformed(ActionEvent ae) {
		Common.startTime = new Date().getTime();
		messageTextArea.setText("Processing ... Please wait.\r\n");
		int tempRepeatTimes = repeatTimesField.getValue();

		// Parameters to be transferred to respective objects.
		String tempFilename = arffFilenameField.getText().trim();
		int tempDistanceMeasure = distanceJComboBox.getSelectedIndex();
		boolean tempRetrospect = retrospectCheckbox.getState();
		boolean tempNormalize = normalizeCheckbox.getState();
		boolean tempDisorder = disorderCheckbox.getState();
		double tempAdaptiveRatio = adaptiveRatioDoubleField.getValue();
		int tempSmallBlockThreshold = smallBlockThresholdIntegerField
				.getValue();
		int tempKValue = kValueIntegerField.getValue();
		int tempInstanceSelectionStrategy = selectCriticalStrategyJComboBox
				.getSelectedIndex();
		int tempQueryAmountStrategy = queryAmountStrategyJComboBox
				.getSelectedIndex();
		double tempNeighborBasedWeight = neighborBasedWeightDoubleField
				.getValue();

		SimpleTools.processTracking = processTrackingCheckbox.getState();
		SimpleTools.variableTracking = variableTrackingCheckbox.getState();
		SimpleTools.fileOutput = fileOutputCheckbox.getState();

		int tempEnsembleAlgorithm = ensembleAlgorithmJComboBox
				.getSelectedIndex();

		ClusteringAlgorithmsBasedActiveLearning tempCenal = null;
		String resultMessage = "";

		String tempOutputFilename = "src/results/" + new Date().getTime()
				+ ".txt";
		RandomAccessFile tempOutputFile = null;
		if (SimpleTools.fileOutput) {
			try {
				tempOutputFile = new RandomAccessFile(tempOutputFilename, "rw");
			} catch (Exception ee) {
				System.out.println(ee);
				System.exit(0);
			} // Of try
		}// Of if

		boolean[] tempAvailableAlgorithms = new boolean[ClusteringAlgorithmSelectionActiveLearning.NUM_ALGORITHMS];
		for (int i = 0; i < tempAvailableAlgorithms.length; i++) {
			tempAvailableAlgorithms[i] = algorithmCheckboxes[i].getState();
		} // Of for i

		switch (tempEnsembleAlgorithm) {
		case 0:
			if (indicateQueriesCheckbox.getState()) {
				double tempQueryFraction = queryFractionField.getValue();
				double tempRepresentativeFraction = representativeFractionField
						.getValue();
				tempCenal = new SpecifiedLabelsAlgorithmSelection(tempFilename,
						tempDistanceMeasure, tempNormalize, tempDisorder,
						tempAdaptiveRatio, tempSmallBlockThreshold,
						tempInstanceSelectionStrategy, tempQueryAmountStrategy,
						tempNeighborBasedWeight, tempQueryFraction,
						tempRepresentativeFraction, tempRetrospect);
			} else {
				tempCenal = new ClusteringAlgorithmSelectionActiveLearning(
						tempFilename, tempDistanceMeasure, tempNormalize,
						tempDisorder, tempAdaptiveRatio,
						tempSmallBlockThreshold, tempInstanceSelectionStrategy,
						tempQueryAmountStrategy, tempNeighborBasedWeight);
			} // Of if
			break;
		case 1:
			tempCenal = new ClusterEnsembleActiveLearning(tempFilename,
					tempDistanceMeasure, tempNormalize, tempDisorder,
					tempAdaptiveRatio, tempSmallBlockThreshold,
					tempInstanceSelectionStrategy, tempQueryAmountStrategy);
			break;
		default:
			System.out
					.println("Internal error occurred in Cenal, the algorithm #"
							+ tempEnsembleAlgorithm + " is not supported.");
			System.exit(0);
		}// Of switch

		tempCenal.setAvailableAlgorithms(tempAvailableAlgorithms);
		tempCenal.setKValue(tempKValue);

		int[] tempQueriesArray = new int[tempRepeatTimes];
		double tempTotalQueries = 0;
		int tempMaxQueries = Integer.MIN_VALUE;
		int tempMinQueries = Integer.MAX_VALUE;

		double[] tempAccuracyArray = new double[tempRepeatTimes];
		double tempAccuracySum = 0;
		double tempMaxAccuracy = -1;
		double tempMinAccuracy = 1.1;

		double[] tempAlgorithmAverageWinArray = new double[ClusteringAlgorithmSelectionActiveLearning.NUM_ALGORITHMS];
		int[] tempAlgorithmWinArray;

		double tempTotalBlocks = 0;
		int tempBlocks;
		int tempMaxBlocks = Integer.MIN_VALUE;
		int tempMinBlocks = Integer.MAX_VALUE;

		messageTextArea.append(resultMessage);

		String tempCurrentRunMessage = "No., Queries,";
		messageTextArea.append(tempCurrentRunMessage + "\r\n");

		for (int i = 0; i < tempRepeatTimes; i++) {
			tempCurrentRunMessage += "\r\n" + (i + 1) + ",";
			// Attention: this approach saves times and space!
			SimpleTools.processTrackingOutput("Reset for the " + i
					+ "th run.\r\n");
			tempCenal.reset();

			tempCurrentRunMessage = tempCenal.learn();
			// System.out.println(tempCurrentRunMessage);
			resultMessage += tempCurrentRunMessage + "\r\n";

			/*
			 * // With in block 1NN, this situation rarely happens. int
			 * tempUnhandled = tempCenal
			 * .getNumInstancesByStatus(ActiveLearning.UNHANDLED); if
			 * (tempUnhandled > 0) { System.out
			 * .println("\r\nUnhandled before kNN: " + tempUnhandled);
			 * resultMessage += "\r\nUnhandled before kNN: " + tempUnhandled +
			 * "\r\n"; // Classify unhandled instances using kNN.
			 * tempCenal.knnUnhandled(3); } // Of if
			 */

			tempQueriesArray[i] = tempCenal.getNumQueries();
			if (tempMaxQueries < tempQueriesArray[i]) {
				tempMaxQueries = tempQueriesArray[i];
			} // Of if
			if (tempMinQueries > tempQueriesArray[i]) {
				tempMinQueries = tempQueriesArray[i];
			} // Of if

			tempAccuracyArray[i] = tempCenal.computeAccuracy();
			if (tempMaxAccuracy < tempAccuracyArray[i]) {
				tempMaxAccuracy = tempAccuracyArray[i];
			} // Of if
			if (tempMinAccuracy > tempAccuracyArray[i]) {
				tempMinAccuracy = tempAccuracyArray[i];
			} // Of if

			tempBlocks = tempCenal.getNumBlocks();
			tempTotalBlocks += tempBlocks;
			if (tempMaxBlocks < tempBlocks) {
				tempMaxBlocks = tempBlocks;
			} // Of if
			if (tempMinBlocks > tempBlocks) {
				tempMinBlocks = tempBlocks;
			} // Of if

			tempAlgorithmWinArray = tempCenal.getAlgorithmWinArray();
			tempTotalQueries += tempQueriesArray[i];
			tempAccuracySum += tempAccuracyArray[i];
			for (int j = 0; j < ClusteringAlgorithmSelectionActiveLearning.NUM_ALGORITHMS; j++) {
				tempAlgorithmAverageWinArray[j] += (tempAlgorithmWinArray[j] + 0.0)
						/ tempRepeatTimes;
			} // Of for j

			messageTextArea.append("\r\n[" + i + "] ");
			messageTextArea.append(tempCurrentRunMessage);
			if (SimpleTools.fileOutput) {
				try {
					tempOutputFile.writeBytes(tempCurrentRunMessage);
				} catch (Exception ee) {
					System.out.println(ee);
					System.exit(0);
				} // Of try
			}// Of if
			tempCurrentRunMessage = "";
		} // Of for i

		if (SimpleTools.fileOutput) {
			try {
				tempOutputFile.close();
			} catch (Exception ee) {
				System.out.println(ee);
				System.exit(0);
			} // Of try
		}// Of if

		double tempAverageAccuracy = tempAccuracySum / tempRepeatTimes;
		double tempStandardDeviation = 0;
		double tempDifference = 0;
		for (int i = 0; i < tempRepeatTimes; i++) {
			tempDifference = tempAccuracyArray[i] - tempAverageAccuracy;
			tempStandardDeviation += tempDifference * tempDifference;
		} // Of for i
		tempStandardDeviation /= tempRepeatTimes;
		tempStandardDeviation = Math.sqrt(tempStandardDeviation);

		messageTextArea.append("\r\nSummary:\r\n");
		messageTextArea.append("Average queries: "
				+ (tempTotalQueries / tempRepeatTimes));
		messageTextArea.append("; Min: " + tempMinQueries);
		messageTextArea.append("; Max " + tempMaxQueries + "\r\n");

		messageTextArea.append("Average accuracy: " + tempAverageAccuracy);
		messageTextArea.append("; Min: " + tempMinAccuracy);
		messageTextArea.append("; Max: " + tempMaxAccuracy + "\r\n");
		messageTextArea.append("; +- " + tempStandardDeviation + "\r\n");

		messageTextArea.append("Average number of blocks: "
				+ (tempTotalBlocks / tempRepeatTimes));
		messageTextArea.append("; Min: " + tempMinBlocks);
		messageTextArea.append("; Max: " + tempMaxBlocks + "\r\n");

		messageTextArea.append("Algorithm win array: "
				+ Arrays.toString(tempAlgorithmAverageWinArray) + "\r\n");

		// String resultMessage = tempCenal.learn();
		// resultMessage += "\r\nThe distance index is: " +
		// tempCenal.getDistanceMeasure();

		Common.endTime = new Date().getTime();
		long tempTimeUsed = Common.endTime - Common.startTime;
		messageTextArea.append("Runtime: " + tempTimeUsed + "\r\n");

		messageTextArea.append("\r\nEnd.");
	}// Of actionPerformed

	/**
	 *************************** 
	 * When the checkbox is selected or deselected.
	 *************************** 
	 */
	public void itemStateChanged(ItemEvent paraEvent) {
		if (paraEvent.getStateChange() == ItemEvent.DESELECTED) {
			queryFractionField.setEnabled(false);
			representativeFractionField.setEnabled(false);
		} else {
			queryFractionField.setEnabled(true);
			representativeFractionField.setEnabled(true);
		} // Of if
	}// Of itemStateChanged

	/**
	 *************************** 
	 * The entrance method.
	 * 
	 * @param args
	 *            The parameters.
	 *************************** 
	 */
	public static void main(String args[]) {
		new Tacs();
	}// Of main
}// Of class Cenal
