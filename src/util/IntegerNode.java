package util;

/**
 * Integer node for building linked lists.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com <br>
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
