import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Main {

	private BufferedReader br;
	private String fileName = "mushrooms.csv";
	private int numberOfTestData = 6000;
	private ArrayList<int[]> trainData = new ArrayList<>();
	private ArrayList<int[]> testData = new ArrayList<>();
	private ArrayList<double[]> normalizedTrainData = new ArrayList<>();
	private ArrayList<double[]> normalizedTestData = new ArrayList<>();

	public Main() {
		try {
			br = new BufferedReader(new FileReader(fileName));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	private void init() {
		System.out.println("Reading data...");
		try {
			boolean skipFirstLine = true;
			int lineCount = 0;
			String line;
			while (br.ready()) {
				line = br.readLine();
				if (skipFirstLine) {
					skipFirstLine = false;
					continue;
				}
				// System.out.print(line);
				if (lineCount < numberOfTestData) {
					testData.add(lineToArray(line));
					// System.out.println(" " + Arrays.toString(testData.get(testData.size() - 1)));
				} else {
					trainData.add(lineToArray(line));
					// System.out.println(" " + Arrays.toString(trainData.get(trainData.size() -
					// 1)));
				}
				lineCount++;
			}
			System.out.println("Normalizing data...");
			for (int[] d : trainData) {
				double[] norm = new double[d.length];
				for (int i = 0; i < d.length; i++) {
					if (letters[i].length() == 1)
						norm[i] = 0;
					else
						norm[i] = (double) d[i] / (letters[i].length() - 1);
					// System.out.print((int) (norm[i] * 10) / 10.0f + " ");
				}
				// System.out.println();
				normalizedTrainData.add(norm);
			}
			for (int[] d : testData) {
				double[] norm = new double[d.length];
				for (int i = 0; i < d.length; i++) {
					if (letters[i].length() == 1)
						norm[i] = 0;
					else
						norm[i] = (double) d[i] / (letters[i].length() - 1);
					// System.out.print((int) (norm[i] * 10) / 10.0f + " ");
				}
				// System.out.println();
				normalizedTestData.add(norm);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private int[] lineToArray(String line) {
		String[] s = line.split(",");
		int[] r = new int[s.length];

		for (int i = 0; i < s.length; i++) {
			r[i] = toInteger(i, s[i].trim());
		}

		return r;
	}

	private String[] letters = new String[23];

	private int toInteger(int n, String s) {
		if (letters[n] == null)
			letters[n] = "";
		if (letters[n].indexOf(s) == -1)
			letters[n] += s;
		return letters[n].indexOf(s);
	}

	private void fitAndGuess() {
		int total = 0;
		int correct = 0;

		NearestNeighbor knn = new NearestNeighbor();
		knn.fit(trainData);

		total = testData.size();
		for (int i = 0; i < testData.size(); i++)
			if (testData.get(i)[0] == knn.guess(testData.get(i)))
				correct++;

		System.out.println("KNN result: " + (correct * 1.0 / total) * 100 + "%");

		//

		NeuralNet net = new NeuralNet(new int[] { 22, 100, 1 }, false);
		// Normalize
		net.fit(normalizedTrainData, 10);

		correct = 0;
		for (int i = 0; i < testData.size(); i++)
			if (normalizedTestData.get(i)[0] == net.guess(normalizedTestData.get(i)))
				correct++;
		System.out.println("NNet result: " + (correct * 1.0 / total) * 100 + "%");

	}

	public static void main(String[] args) {
		Main main = new Main();
		main.init();
		main.fitAndGuess();
	}
}
