package exception;

/**
 * There is less blocks than required.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Finished.
 * Maybe we should fix it in the future. <br>
 * Written time: July 25, 2019. <br>
 * Last modify time: July 25, 2019.
 */

public class LessBlocksThanRequiredException extends Exception {
	/**
	 ********************
	 * The constructor
	 * 
	 * @param paraMessage
	 *            The message to display.
	 ********************
	 */
	public LessBlocksThanRequiredException(String paraMessage) {
		super(paraMessage);
	}// Of the constructor

}// Of class LessBlocksThanRequiredException
