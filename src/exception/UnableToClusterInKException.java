package exception;

public class UnableToClusterInKException extends Exception {
	/**
	 ********************
	 * The constructor
	 * 
	 * @param paraMessage
	 *            The message to display.
	 ********************
	 */
	public UnableToClusterInKException(String paraMessage) {
		super(paraMessage);
	}// Of the constructor

	/**
	 ********************
	 * The constructor
	 * 
	 * @param paraMessage
	 *            The message to display.
	 * @param paraK
	 *            The number of clusters.
	 ********************
	 */
	public UnableToClusterInKException(String paraMessage, int paraK) {
		super(paraMessage + ", k = " + paraK);
	}// Of the constructor

}// Of class UnableToClusterInKException
