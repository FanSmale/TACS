package matrix;

/**
 * A triple for compressed matrix.
 */
public class Triple {
	/**
	 * The column of the node. The row is already given outside this object.
	 */
	public int column;

	/**
	 * The weight of the node.
	 */
	public double weight;

	/**
	 * The next triple. They are organized as a list.
	 */
	public Triple next;

	public Triple() {
		column = 0;
		weight = 0;
		next = null;
	}// Of the constructor

	/**
	 ********************
	 * Construct a new triple.
	 * 
	 * @param paraColumn
	 *            The given column.
	 * @param paraWeight
	 *            The weight.
	 * @param paraNext
	 *            The reference to the next triple.
	 ********************
	 */
	public Triple(int paraColumn, double paraWeight, Triple paraNext) {
		column = paraColumn;
		weight = paraWeight;
		next = paraNext;
	}// Of the constructor

	/**
	 *********************
	 * Inner product of two vectors.
	 * 
	 * @param paraHeader1
	 *            The header of the first array.
	 * @param paraHeader2
	 *            The header of the second array.
	 * @return The multiplex as a double value.
	 *********************
	 */
	public static double multiply(Triple paraHeader1, Triple paraHeader2) {
		double tempWeightSum = 0;

		Triple tempTriple1 = paraHeader1.next;
		Triple tempTriple2 = paraHeader2.next;

		while ((tempTriple1 != null) && (tempTriple2 != null)) {
			if (tempTriple1.column < tempTriple2.column) {
				tempTriple1 = tempTriple1.next;
			} else if (tempTriple2.column < tempTriple1.column) {
				tempTriple2 = tempTriple2.next;
			} else {
				tempWeightSum += tempTriple1.weight * tempTriple2.weight;
				tempTriple1 = tempTriple1.next;
				tempTriple2 = tempTriple2.next;
			} // Of if
		} // Of while

		return tempWeightSum;
	}// Of multiply

	/**
	 *********************
	 * add two arrays.
	 * @param paraHeader1
	 *            The header of the first array.
	 * @param paraHeader2
	 *            The header of the second array.
	 * @return The addition as a new triple.
	 *********************
	 */
	public static Triple add(Triple paraHeader1, Triple paraHeader2) {
		Triple resultHeader = new Triple();
		Triple tempTail = resultHeader;
		Triple tempTriple;

		Triple tempTriple1 = paraHeader1.next;
		Triple tempTriple2 = paraHeader2.next;

		while ((tempTriple1 != null) && (tempTriple2 != null)) {
			if (tempTriple1.column < tempTriple2.column) {
				// Copy the triple of the first array
				tempTriple = new Triple();
				tempTriple.column = tempTriple1.column;
				tempTriple.weight = tempTriple1.weight;

				// Insert to the new array
				tempTail.next = tempTriple;
				tempTail = tempTriple;

				tempTriple1 = tempTriple1.next;
			} else if (tempTriple2.column < tempTriple1.column) {
				// Copy the triple of the second array
				tempTriple = new Triple();
				tempTriple.column = tempTriple2.column;
				tempTriple.weight = tempTriple2.weight;

				// Insert to the new array
				tempTail.next = tempTriple;
				tempTail = tempTriple;

				tempTriple2 = tempTriple2.next;
			} else {
				// Compute the sum
				tempTriple = new Triple();
				tempTriple.column = tempTriple1.column;
				tempTriple.weight = tempTriple1.weight + tempTriple2.weight;

				// Insert to the new array
				tempTail.next = tempTriple;
				tempTail = tempTriple;

				tempTriple1 = tempTriple1.next;
				tempTriple2 = tempTriple2.next;
			} // Of if
		} // Of while

		// Copy the remaining part of the first array
		while (tempTriple1 != null) {
			// Copy the triple of the first array
			tempTriple = new Triple();
			tempTriple.column = tempTriple1.column;
			tempTriple.weight = tempTriple1.weight;

			// Insert to the new array
			tempTail.next = tempTriple;
			tempTail = tempTriple;

			tempTriple1 = tempTriple1.next;
		} // Of while

		// Copy the remaining part of the second array
		while (tempTriple2 != null) {
			// Copy the triple of the first array
			tempTriple = new Triple();
			tempTriple.column = tempTriple2.column;
			tempTriple.weight = tempTriple2.weight;

			// Insert to the new array
			tempTail.next = tempTriple;
			tempTail = tempTriple;

			tempTriple2 = tempTriple2.next;
		} // Of while

		return resultHeader.next;
	}// Of add

	/**
	 *********************
	 * The manhattan distance between two arrays.
	 * @param paraHeader1
	 *            The header of the first array.
	 * @param paraHeader2
	 *            The header of the second array.
	 *            @return The distance.
	 *********************
	 */
	public static double manhattan(Triple paraHeader1, Triple paraHeader2) {
		double resultValue = 0;

		Triple tempTriple1 = paraHeader1.next;
		Triple tempTriple2 = paraHeader2.next;

		while ((tempTriple1 != null) && (tempTriple2 != null)) {
			if (tempTriple1.column < tempTriple2.column) {
				resultValue += tempTriple1.weight;

				tempTriple1 = tempTriple1.next;
			} else if (tempTriple2.column < tempTriple1.column) {
				resultValue += tempTriple2.weight;

				tempTriple2 = tempTriple2.next;
			} else {
				resultValue += Math.abs(tempTriple1.weight - tempTriple2.weight);
				// Compute the sum

				tempTriple1 = tempTriple1.next;
				tempTriple2 = tempTriple2.next;
			} // Of if
		} // Of while

		// Add the remaining part of the first array
		while (tempTriple1 != null) {
			resultValue += tempTriple1.weight;

			tempTriple1 = tempTriple1.next;
		} // Of while

		// Add the remaining part of the second array
		while (tempTriple2 != null) {
			resultValue += tempTriple2.weight;

			tempTriple2 = tempTriple2.next;
		} // Of while

		return resultValue;
	}// Of manhattan

	/**
	 *********************
	 * Display it.
	 * @return The string displaying it.
	 *********************
	 */
	public String toString() {
		String resultString = "";
		Triple tempReference = this;
		while (tempReference != null) {
			resultString += tempReference.column;
			resultString += ", ";
			resultString += tempReference.weight;
			resultString += "; ";
			tempReference = tempReference.next;
		} // Of while
		return resultString;
	}// Of toString
}// Of triple
