package common;

/**
 * Integer node for building linked lists.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Finished.<br>
 * Written time: October 20, 2017. <br>
 * Last modify time: July 27, 2019.
 */

public class IntegerNode {

	/**
	 * The value. Usually an index of the data item.
	 */
	public int value;

	/**
	 * The value. Usually an index of the data item.
	 */
	public IntegerNode next;

	/**
	 ************************* 
	 * The constructor.
	 * 
	 * @param paraValue
	 *            The initial value of the node.
	 ************************* 
	 */
	public IntegerNode(int paraValue) {
		value = paraValue;
		next = null;
	}// Of the constructor

	/**
	 ************************* 
	 * Set the next node.
	 * 
	 * @param paraNext
	 *            The next node.
	 ************************* 
	 */
	public void setNext(IntegerNode paraNext) {
		next = paraNext;
	}// Of the constructor
}// Of class
