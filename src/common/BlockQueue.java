package common;

/**
 * Clustering-based active learning. A queue managing blocks.
 * <p>
 * Author: <b>Fan Min</b> minfanphd@163.com, minfan@swpu.edu.cn <br>
 * Copyright: The source code and all documents are open and free. PLEASE keep
 * this header while revising the program. <br>
 * Organization: <a href=http://www.fansmale.com/>Lab of Machine Learning</a>,
 * Southwest Petroleum University, Chengdu 610500, China.<br>
 * Project: The cost-sensitive active learning project.
 * <p>
 * Progress: Almost finished, further revision is possible.<br>
 * Written time: July 21, 2019. <br>
 * Last modify time: July 21, 2019.
 */

public class BlockQueue {
	/**
	 * The default queue length.
	 */
	public static final int DEFAULT_QUEUE_LENGTH = 1000;

	/**
	 * The current queue. It is a queue of blocks.
	 */
	int[][] currentQueue;

	/**
	 * The length of the queue. It should not be exceeded.
	 */
	int queueLength;

	/**
	 * The head.
	 */
	int head;

	/**
	 * The tail.
	 */
	int tail;

	/**
	 ****************** 
	 * The first constructor.
	 ****************** 
	 */
	public BlockQueue() {
		queueLength = DEFAULT_QUEUE_LENGTH;
		currentQueue = new int[queueLength][];
		head = 0;
		tail = 0;
	}// Of the constructor

	/**
	 ****************** 
	 * Enqueue.
	 * 
	 * @param paraBlock
	 *            The block to enqueue.
	 ****************** 
	 */
	public void enqueue(int[] paraBlock) {
		SimpleTools.consoleOutput("Enqueue a block with " + paraBlock.length + " instances");
		currentQueue[head] = paraBlock;
		head++;
		if (head == currentQueue.length) {
			System.out.println(
					"Fatal error occurred in SpecifiedLabelsAlgorithmSelection.enqueue(): no enough space.");
			System.exit(0);
		} // Of if
	}// Of enqueue

	/**
	 ****************** 
	 * Dequeue.
	 * 
	 * @return The block to at the tail.
	 ****************** 
	 */
	public int[] dequeue() {
		tail++;
		if (tail > head) {
			System.out.println("Fatal error occurred in SpecifiedLabelsAlgorithmSelection.dequeue(): " + "head = "
					+ head + " and tail = " + tail + ". No element to take out.");
			System.exit(0);
		} // Of if

		SimpleTools.consoleOutput("dequeue a block with " + currentQueue[tail - 1].length + " instances");
		return currentQueue[tail - 1];
	}// Of dequeue

	/**
	 ****************** 
	 * Get the actual length of the queue.
	 * 
	 * @return The length of the queue.
	 ****************** 
	 */
	public int getLength() {
		return head - tail;
	}//Of getLength
	
	/**
	 ****************** 
	 * Is the queue empty.
	 * 
	 * @return True if empty.
	 ****************** 
	 */
	public boolean isEmpty() {
		if (head == tail) {
			return true;
		} // Of if

		return false;
	}// Of dequeue
}// Of class BlockQueue
