import java.util.ArrayList;

public class NearestNeighbor {

	private ArrayList<int[]> data = new ArrayList<>();

	public NearestNeighbor() {
		// Nearest k?
	}

	public void fit(ArrayList<int[]> data) {
		this.data = data;
	}

	public int guess(int[] testData) {
		int nearestIndex = 0;
		double nearestDistance = distance(data.get(0), testData);
		for (int i = 1; i < data.size(); i++) {
			double dist = distance(data.get(i), testData);
			if (dist < nearestDistance) {
				nearestDistance = dist;
				nearestIndex = i;
			}
		}
		return data.get(nearestIndex)[0];
		// return data.get(new Random().nextInt(data.size()))[0];
	}

	private double distance(int[] d1, int[] d2) {
		double r = 0;
		for (int i = 1; i < d1.length; i++)
			r += (d1[i] - d2[i]) * (d1[i] - d2[i]);
		r = Math.sqrt(r);
		return r;
	}
}
