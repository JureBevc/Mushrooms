import java.util.ArrayList;

public class NeuralNet {

	private Node nodes[][];
	private boolean bias; // Bias nodes

	public NeuralNet(int nodeNumbers[], boolean bias) {
		this.bias = bias;
		nodes = new Node[nodeNumbers.length][];
		for (int i = 0; i < nodeNumbers.length; i++) {
			nodes[i] = new Node[nodeNumbers[i] + (bias ? 1 : 0)]; // +1 for biased nodes
			for (int j = 0; j < nodes[i].length; j++)
				nodes[i][j] = new Node((i == 0 ? 0 : nodes[i - 1].length), j);
			if (bias)
				nodes[i][nodes[i].length - 1].output = 1; // biased node output
		}
	}

	public void feedInput(double inputValues[]) {
		// Input layer
		if (inputValues.length != nodes[0].length - (bias ? 1 : 0))
			System.out.println("Input values don't match the number of input nodes!");

		for (int i = 0; i < inputValues.length; i++)
			nodes[0][i].output = inputValues[i];

		// Forward propagation
		for (int i = 1; i < nodes.length; i++) {
			for (int j = 0; j < nodes[i].length - (bias ? 1 : 0); j++) {
				nodes[i][j].feed(nodes[i - 1]);
			}
		}
	}

	public void backProp(double targetValue[]) {

		// Calculate error
		// RMS? -> sqrt(sum(delta^2) / n)
		double err = 0;
		for (int i = 0; i < targetValue.length; i++) {
			double delta = targetValue[i] - nodes[nodes.length - 1][i].output;
			err += delta * delta;
		}
		err = Math.sqrt(err / targetValue.length); // RMS

		// Output layer gradient
		for (int i = 0; i < targetValue.length; i++) {
			nodes[nodes.length - 1][i].calcOutputGradient(targetValue[i]);
		}

		// Hidden layer gradients
		for (int i = nodes.length - 2; i > 0; i--) {
			for (int j = 0; j < nodes[i].length - (bias ? 1 : 0); j++) {
				nodes[i][j].calcHiddenGradient(nodes[i + 1], bias);
			}
		}

		// Update weights
		for (int i = nodes.length - 1; i > 0; i--) {
			for (int j = 0; j < nodes[i].length - (bias ? 1 : 0); j++) {
				nodes[i][j].updateWeights(nodes[i - 1]);
			}
		}
	}

	public void fit(ArrayList<double[]> normalizedData, int loops) {
		double[] input = new double[nodes[0].length - (bias ? 1 : 0)];
		double[] output = new double[nodes[nodes.length - 1].length - (bias ? 1 : 0)];
		for (int loop = 0; loop < loops; loop++) {
			for (double[] d : normalizedData) {
				for (int i = 0; i < input.length; i++)
					input[i] = d[i + 1];
				output[0] = d[0];
				feedInput(input);
				backProp(output);
			}
		}

	}

	public int guess(double[] normalizedTestData) {
		double[] input = new double[nodes[0].length - (bias ? 1 : 0)];
		double[] output = new double[nodes[nodes.length - 1].length - (bias ? 1 : 0)];
		for (int i = 0; i < input.length; i++)
			input[i] = normalizedTestData[i + 1];
		output[0] = normalizedTestData[0];
		feedInput(input);

		double r = nodes[nodes.length - 1][0].output;
		return (r >= 0.5 ? 1 : 0);
	}
}

class Node {

	public double output = 0;
	public double weights[];
	public double deltaWeights[];
	public double grad = 0;
	private int id; // Position in layer

	public Node(int numberOfInputs, int id) {
		this.id = id;
		weights = new double[numberOfInputs];
		deltaWeights = new double[numberOfInputs];

		// Random weights
		for (int i = 0; i < weights.length; i++)
			weights[i] = Math.random(); // Math.random() * 2 - 1 ????
	}

	public double act(double x) {
		// Activator function
		return Math.tanh(x);
	}

	public double actDerivative(double x) {
		double r = Math.tanh(x);
		return 1 - r * r;
	}

	public void feed(Node[] lastLayer) {
		// output = activator(sum(input * weights))
		double sum = 0;
		for (int i = 0; i < lastLayer.length; i++)
			sum += lastLayer[i].output * weights[i];
		output = act(sum);
	}

	public void updateWeights(Node[] lastLayer) {
		// eta -> learn rate
		// alpha -> momentum
		double eta = 0.01;
		double alpha = 0.5;
		for (int i = 0; i < lastLayer.length; i++) {
			double newDelta = eta * lastLayer[i].output * grad + alpha * deltaWeights[i];
			deltaWeights[i] = newDelta;
			weights[i] += deltaWeights[i];
		}
	}

	public void calcOutputGradient(double targetValue) {
		// For calculating the gradient of output (last layer) nodes
		double delta = targetValue - output;
		grad = delta * actDerivative(output);
	}

	public void calcHiddenGradient(Node[] nextLayer, boolean bias) {
		// For calculating the gradient of all hidden nodes
		double dow = 0;
		// Sum of the derivatives of weights of next layer
		for (int i = 0; i < nextLayer.length - (bias ? 1 : 0); i++) {
			dow += nextLayer[i].weights[id] * nextLayer[i].grad;
		}
		//
		grad = dow * actDerivative(output);
	}

}