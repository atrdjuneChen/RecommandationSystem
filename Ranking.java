package CollaborativeRanking;

import java.util.*;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import java.io.*;

public class Ranking {

	int feature_dimension = 30;
	long pair_wise_training_num;  // total number of pair wise training data
	
	HashMap<Integer, Vector> User;
	HashMap<Integer, Vector> Item;
	HashSet<Integer> valid_user_id;
	HashSet<Integer> valid_item_id;
	HashMap<Integer,HashMap<Integer,Integer>> map;
	
	public Ranking (HashMap<Integer,HashMap<Integer,Integer>> map, HashSet<Integer> valid_user_id, HashSet<Integer> valid_item_id)
	{
		this.User=new HashMap<Integer, Vector>();
		this.Item=new HashMap<Integer, Vector>();
		this.valid_item_id = valid_item_id;
		this.valid_user_id = valid_user_id;
		this.map = map;
	}
	
	public void loadingFeature(String userFeatureFilePath, String itemFeatureFilePath) throws IOException
	{
		System.out.println("loading feature");
		File userFeatureFile = new File(userFeatureFilePath);
		File itemFeatureFile = new File(itemFeatureFilePath);
		BufferedReader reader=null;
		String str = null;
		for(int k=0; k < 2; k++)
		{
			if(k==0)
				reader = new BufferedReader(new FileReader(userFeatureFile));
			else
				reader = new BufferedReader(new FileReader(itemFeatureFile));
			while((str=reader.readLine())!=null)
			{
				String[] inputs = str.split(" ");
				
				int id=Integer.valueOf(inputs[0]);
				if(inputs[1].equals("{}"))
					continue;
				
				String feature_vector = inputs[1].substring(1, inputs[1].length()-1);
				String[] temp = feature_vector.split(",");
				Vector v = new DenseVector(this.feature_dimension);
				for(String pair : temp)
				{
					String[] entry = pair.split(":");
					int pos = Integer.valueOf(entry[0]);
					double value = Double.valueOf(entry[1]);
					v.set(pos, value);
				}
				if(k==0)
					this.User.put(id, v);
				if(k==1)
					this.Item.put(id, v);
			}
		}
		reader.close();
		System.out.println("----> finished\n");
	}
	
	public void buildingPairFeatureVectors_train(String training_data_outputFilePath) throws IOException
	{
		System.out.println("building pair wise feature vector for training");
		File f = new File(training_data_outputFilePath);
		BufferedWriter writer = new BufferedWriter(new FileWriter(f));
		StringBuilder str = null;
		long totalfeature_num = 0;
		// for each existing users, find out its corresponding items which is 1 star or 5 star
		for(int userId : this.valid_user_id)
		{
			LinkedList<Integer> star5 = new LinkedList<Integer>();
			LinkedList<Integer> star1 = new LinkedList<Integer>();
			for(Integer itemId : this.map.get(userId).keySet())
			{
				int rating = this.map.get(userId).get(itemId);
				if(rating==1)
					star1.add(itemId);
				else if(rating==5)
					star5.add(itemId);
			}
			if(star5.size()==0 || star1.size()==0)
				continue;
			else
			{
				for(int item_star5 : star5)
				{
					Vector x5 = this.User.get(userId).times(this.Item.get(item_star5));
					for(int item_star1 : star1)
					{
						if(item_star1 < item_star5)
							continue;
						Vector x1 = this.User.get(userId).times(this.Item.get(item_star1));
						Vector v = x5.minus(x1);
						str = new StringBuilder("1 ");
						for(int i=0; i<v.size(); i++)
						{
							str.append(String.valueOf(i+1));
							str.append(":");
							str.append(v.get(i));
							str.append(" ");
						}
						writer.write(str.toString());
						writer.newLine();
						totalfeature_num++;
					}
				}
				for(int item_star1 : star1)
				{
					Vector x1 = this.User.get(userId).times(this.Item.get(item_star1));
					
					for(int item_star5 : star5)
					{
						if(item_star5 < item_star1)
							continue;
						Vector x5 = this.User.get(userId).times(this.Item.get(item_star5));
						Vector v = x1.minus(x5);
						str = new StringBuilder("-1 ");
						for(int i=0; i<v.size(); i++)
						{
							str.append(String.valueOf(i+1));
							str.append(":");
							str.append(v.get(i));
							str.append(" ");
						}
						writer.write(str.toString());
						writer.newLine();
						totalfeature_num++;
					}
				}
			}
		}
		System.out.println("total pair feature number:"+totalfeature_num);
		System.out.println("----> finished\n");
		writer.close();
		this.pair_wise_training_num = totalfeature_num;
	}
	
	
	public void buildingPairFeatureVectors_test(String testDataInputFilePath, String testingFeatureVectorsFile)throws IOException
	{
		System.out.println("building pair wise feature vector for testing");
		File fin = new File(testDataInputFilePath);
		File fout= new File(testingFeatureVectorsFile);
		BufferedReader reader = new BufferedReader(new FileReader(fin));
		BufferedWriter writer = new BufferedWriter(new FileWriter(fout));
		StringBuilder sb = null;
		String str = null;
		while((str = reader.readLine())!=null)
		{
			String[] temp = str.split(",");
			int item = Integer.valueOf(temp[0]);
			int user = Integer.valueOf(temp[1]);
			Vector v = this.Item.get(item).times(this.User.get(user));
			sb = new StringBuilder("0 ");
			for(int i=0; i<v.size(); i++)
			{
				sb.append(String.valueOf(i+1));
				sb.append(":");
				sb.append(v.get(i));
				sb.append(" ");
			}
			writer.write(sb.toString());
			writer.newLine();
		}
		writer.close();
		reader.close();
		System.out.println("----> finished\n");
	}
	
	public void rank(String classifyOutputFile, String rankingOutputFile, String rankingOutputFile_detailed,
			HashMap<Integer, List<Integer>> queries_per_user, List<String> item_user_pairs_order, String method) throws IOException
	{
		if(method.equals("svm"))
			System.out.println("ranking with svm");
		else
			System.out.println("ranking with lr");
		HashMap<Integer, List<Pair>> scores_per_user = new HashMap<Integer, List<Pair>>();
		
		File fin = new File(classifyOutputFile);
		File fout = new File(rankingOutputFile);
		BufferedReader reader = new BufferedReader(new FileReader(fin));
		BufferedWriter writer= new BufferedWriter(new FileWriter(fout));
		String str = null;
		int i=0;
		while((str = reader.readLine())!=null)
		{
			if(i==0)
			{
				System.out.println(str);
				i++;
				continue;
			}
			// format: label probability of 1, probability of -1
			String[] info = item_user_pairs_order.get(i-1).split(",");
			int user = Integer.valueOf(info[1]);
			if(!scores_per_user.containsKey(user))
				scores_per_user.put(user, new LinkedList<Pair>());
			int item = Integer.valueOf(info[0]);
			String[] temp = str.split(" ");
			double score = Double.valueOf(temp[1]);
			Pair pair = new Pair(item, score);
			scores_per_user.get(user).add(pair);
			writer.write(temp[1]);
			writer.newLine();
			i++;
		}
		for(int user : scores_per_user.keySet())
			Collections.sort(scores_per_user.get(user));
		reader.close();
		writer.close();
		
		File fout_detailed = new File(rankingOutputFile_detailed);
		writer = new BufferedWriter(new FileWriter(fout_detailed));
		StringBuilder sb = null;
		writer.write("userID (itemID:RankingScore)");
		writer.newLine();
		for(int user : scores_per_user.keySet())
		{
			List<Pair> ranklist = scores_per_user.get(user);
			sb = new StringBuilder(String.valueOf(user)+" ");
			for(Pair p : ranklist)
				sb.append("("+p.get_itemID()+":"+p.get_score()+"), ");
			
			writer.write(sb.toString().substring(0,sb.length()-1));
			writer.newLine();
		}
		writer.close();
		System.out.println("----> finished\n");
	}
	
	
}

class Pair implements Comparable<Pair>
{
	int itemID;
	double score;
	public Pair(int itemID, double score)
	{
		this.itemID = itemID;
		this.score = score;
	}
	public int compareTo(Pair o) {
		return Double.compare(o.score, this.score);   // descending order
	}
	public int get_itemID(){
		return this.itemID;
	}
	public double get_score(){
		return this.score;
	}
}
