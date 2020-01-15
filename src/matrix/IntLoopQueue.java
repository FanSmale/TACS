package matrix;

public class IntLoopQueue {
	int size;
	int[] data;
	int head;
	int tail;

	/**
	 *********************
	 * Construct a loop queue.
	 * 
	 * @param paraSize
	 *            the size of the queue
	 *********************
	 */
	public IntLoopQueue(int paraSize) {
		size = paraSize;
		data = new int[size];
		head = 0;
		tail = 0;
	}// Of the first constructor

	/**
	 *********************
	 * Reset the queue.
	 *********************
	 */
	public void reset() {
		head = 0;
		tail = 0;
	}// Of reset

	/**
	 *********************
	 * Add an element to the tail of the queue
	 * 
	 * @param paraData
	 *            The enqueued data.
	 * @throws Exception
	 *             If the queue is full.
	 *********************
	 */
	public void enqueue(int paraData) throws Exception {
		if ((tail + 1) % size == head) {
			throw new Exception("Int queue is full, cannot enqueue for " + paraData);
		} // Of if

		data[tail] = paraData;
		tail = (tail + 1) % size;
	}// Of enqueue

	/**
	 *********************
	 * Is the queue empty?
	 * 
	 * @return Empty or not.
	 *********************
	 */
	public boolean isEmpty() {
		if (tail == head) {
			return true;
		}
		return false;
	}// Of isEmpty

	/**
	 *********************
	 * Get an element from the queue
	 * 
	 * @return The element taken out of the queue.
	 * @throws Exception
	 *             for empty queue.
	 *********************
	 */
	public int dequeue() throws Exception {
		if (isEmpty()) {
			throw new Exception("Int queue is empty, cannot dequeue!");
		} // Of if

		int tempResult = data[head];
		head = (head + 1) % size;
		return tempResult;
	}// Of dequeue

	/**
	 *********************
	 * Test the queue.
	 * 
	 * @param args
	 *            The parameters.
	 *********************
	 */
	public static void main(String args[]) {
		IntLoopQueue tempQueue = new IntLoopQueue(4);
		try {
			tempQueue.enqueue(1);
			tempQueue.enqueue(2);
			tempQueue.enqueue(3);

			System.out.println("Get " + tempQueue.dequeue());
			tempQueue.enqueue(4);
			System.out.println("Get " + tempQueue.dequeue());
			System.out.println("Get " + tempQueue.dequeue());
			System.out.println("Get " + tempQueue.dequeue());
			tempQueue.enqueue(5);
			tempQueue.enqueue(6);
			System.out.println("Get " + tempQueue.dequeue());
			System.out.println("Get " + tempQueue.dequeue());
			System.out.println("Get " + tempQueue.dequeue());
			System.out.println("Get " + tempQueue.dequeue());
		} catch (Exception ee) {
			System.out.println("Error occurred while testing IntLoopQueue: " + ee);
		}
	}// Of main

}// Of class IntLoopQueue
