package exception;

import java.lang.Exception;

public class DuplicateQueryException extends Exception {
	/**
	 ********************
	 * The constructor
	 * 
	 * @param paraMessage
	 *            The message to display.
	 ********************
	 */
	public DuplicateQueryException(String paraMessage) {
		super(paraMessage);
	}// Of the constructor

}// Of class DuplicateQueryException
