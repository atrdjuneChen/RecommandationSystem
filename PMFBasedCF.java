package CollaborativeRanking;

import java.util.*;
import java.util.Arrays;
import java.io.*;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import com.google.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import org.apache.mahout.math.function.Functions;

/**
* Probabilistic Matrix Factorization. 
* Factorize a data matrix into two matrices
* Using gradient descent
* 
* ps. ratings values are preprocess in the Driver class, 
* using (x-1)/(K-1)
* now the rating values scale from [0,1]
*/

public class PMFBasedCF {
	
	private Matrix R;  //(n,m)
	private Matrix P;  //(n,r)   user matrix
	private Matrix Q;  //(r,m)   transpose of item matrix
	private Matrix Rating_records;  // if user i rates item j, Rating_records(i,j)=1
	private int n;  // row number of R
	private int m;  // column number of R
	private int r;  // number of latent features
	private double obective_value; // value of the objective function
	
	// hyperameters
	private double beta1=0.001;  // hyperameter for P
	private double beta2=0.001;  // hyperameter for Q
	
	private int maxStepNum = 200;  // maximum iteration times
	double threshold = 0.01;   // maximum error
	double alpha =  0.0001;     // learning rate for gradient descent
	
	int K=5;     // highest rating value, starting from 1
	int user_num;
	int item_num;
	HashSet<Integer> valid_user_id;
	HashSet<Integer> valid_item_id;
	int[] valid_user;   // the array form of HashSet
	int[] valid_item;
	Vector average_rating_vector;   // average rating of each (recorded) item, used for cold start of unknown user.
	double average_rating;          // average rating of an unknown item, used for cold start of unknown item.
	
	HashMap<Integer,HashMap<Integer,Integer>> map;  // HashMap form of Rating Matrix R
	HashMap<Integer, HashMap<Integer, Double>> mini_validation_set ;  // mini validation set which is used during training.
	
	
	public PMFBasedCF(Matrix R, int K, Matrix Rating_records,  HashSet<Integer> valid_user_id, HashSet<Integer> valid_item_id, Vector average_rating_vector,
			int number_latenfeature, double learning_rate, double beta1, double beta2, int maxStepNum, double threshold, 
			HashMap<Integer, HashMap<Integer, Double>> MiniValidationSet, HashMap<Integer,HashMap<Integer,Integer>> map)
	{
		System.out.println("\nPMFBased");
		
		this.mini_validation_set = MiniValidationSet;
        this.map = map;
        
		this.R = R;
		this.n = R.rowSize();
	    this.m = R.columnSize();
	    this.r = number_latenfeature;
	    this.alpha = learning_rate;
	    this.maxStepNum = maxStepNum;
	    this.threshold = threshold;
	    
	    this.beta1 = beta1;
        this.beta2 = beta2;
        
        // randomly assign to P and Q 
        this.P = new DenseMatrix(n, r).assign(Functions.random());
        this.Q = new DenseMatrix(r, m).assign(Functions.random());
        this.obective_value = calculate_object_value(R, P, Q);
        
        
        this.K = K;
        this.valid_user_id = valid_user_id;
		this.valid_item_id = valid_item_id;
		this.valid_user = new int[valid_user_id.size()];
		this.valid_item = new int[valid_item_id.size()];
		
		int count = 0;
		for(int user : valid_user_id)
			this.valid_user[count++] = user;
		count = 0;
		for(int item : valid_item_id)
			this.valid_item[count++] = item;
		
		this.average_rating_vector = average_rating_vector;
		this.average_rating = (1.0) * average_rating_vector.zSum() / average_rating_vector.getNumNonZeroElements();
		this.user_num = R.rowSize();
		this.item_num = R.columnSize();
		
		this.Rating_records = Rating_records;
	}
	
	
	
	/**
	 * calculate the value of objective function
	 * */
	public double calculate_object_value (Matrix R, Matrix P, Matrix Q) {
        Matrix PQ = (P.times(Q));
        double err = 0;
        for(int i=0; i<n; i++)
        	for(int j=0; j<m; j++)
        		 if (R.get(i, j) != 0) err += Math.pow(R.get(i, j) - PQ.get(i, j), 2);
        
        for(int i=0; i<n; i++) 
        	for(int j=0; j<r; j++) 
        		err += (beta1) * (Math.pow(P.get(i, j), 2));
        
        for(int i=0; i<r; i++) 
        	for(int j=0; j<m; j++) 
        		err += (beta2) * (Math.pow(Q.get(i, j), 2));
        
        return err/2;
	}
	
	// TODO
	/**
	 * calculate the predicted rating
	 * using logistic function g(x) = 1/(1 + exp(−x)), which bounds the range of predictions
	 * */
	public double predict_rating_logic (Vector v1, Vector v2)
	{
		double ans = 0.0;
		ans = v1.dot(v2);
		return 1.0 / (1.0 + Math.exp(-1.0 * ans));
	}
	
	/**
	 * using gradient descent to do PMF
	 * */
	public void solve () 
	{
		System.out.println("solving PMF using gradient descent");
		int count = 0;
		
		double rmse = Double.MAX_VALUE;
		while(count < maxStepNum)
		{
			// first update matrix P
			for(int i=0; i<this.P.rowSize(); i++)
			{
				if(!this.valid_user_id.contains(i))
					continue;
				Vector gradient_p = new RandomAccessSparseVector(this.r);
				for(int j=0; j<this.Q.columnSize(); j++)
				{
					if(this.R.viewRow(i).get(j)==0)
						continue;
					
					double predict_rating = predict_rating_logic ( P.viewRow(i), Q.viewColumn(j) );
				    double eij = (R.get(i, j) - predict_rating);
				    
				    gradient_p = gradient_p.plus( Q.viewColumn(j).times(-eij) );
				}
				gradient_p = gradient_p.plus(P.viewRow(i).times(this.beta1));    // add regularization term
				this.P.assignRow(i, P.viewRow(i).plus(gradient_p.times(-this.alpha)));
			}
			
			// update matrix Q
			for(int j=0; j<this.Q.columnSize(); j++)
			{
				if(!this.valid_item_id.contains(j))
					continue;
				Vector gradient_q = new RandomAccessSparseVector(this.r);
				for(int i=0; i<this.P.rowSize(); i++)
				{
					if(this.R.viewRow(i).get(j)==0)
						continue;
					double predict_rating = predict_rating_logic(P.viewRow(i),(Q.viewColumn(j)));
					double eij = R.get(i, j) - predict_rating;
					gradient_q = gradient_q.plus( P.viewRow(i).times(-eij));
				}
				gradient_q = gradient_q.plus(Q.viewColumn(j).times(this.beta2));
				this.Q.assignColumn(j, Q.viewColumn(j).plus(gradient_q.times(-this.alpha)));
			}
			
			double rmse_validation_set = predict_validationSet();
			System.out.println("iteration "+ (++count) + "\trmse: " + rmse_validation_set);
			// stop criteria
			if(rmse_validation_set < this.threshold)
				break;
		}
		System.out.println("traning error: " + training_error_rmse());
		
	}
	
	
    public double predict_validationSet()
    {
    	double rmse = 0.0;
    	int count = 0;
    	for(int userId : this.mini_validation_set.keySet())
    	{
    		for(int itemId : this.mini_validation_set.get(userId).keySet())
    		{
    			double rating = this.mini_validation_set.get(userId).get(itemId); // scaled
    			double predict = predict_rating_logic(P.viewRow(userId), Q.viewColumn(itemId));     // scaled, multiply (this.K-1) + 1.0 to [1,K]
    			
    			double r = this.map.get(userId).get(itemId);
    			double p = predict * (this.K-1) + 1.0;
    			rmse += (r-p) * (r-p);
    			count ++;
    		}
    	}
    	return Math.sqrt(rmse/count);
    }
    
    public double training_error_rmse()
    {
    	double rmse = 0.0;
    	int count = 0;
    	for(int userId : this.map.keySet())
    	{
    		for(int itemId : this.map.get(userId).keySet())
    		{
    			int rating = this.map.get(userId).get(itemId);
    			double predict = (predict_rating_logic(P.viewRow(userId), Q.viewColumn(itemId)))*(this.K-1) + 1.0;
    			rmse += (rating-predict) * (rating-predict);
    			count ++;
    		}
    	}
    	return Math.sqrt(rmse/count);
    }
    
    public void predict(String test_file_path, String output_file_path)  throws IOException
	{
		System.out.println("start prediction---");
		
		String outputFilePath_PMF = output_file_path+"//PMFBased_"+this.r+"_.txt";
		File output_PMF = new File(outputFilePath_PMF);
		BufferedWriter writer = new BufferedWriter(new FileWriter(output_PMF));
		
		FileInputStream is=new FileInputStream(test_file_path);
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		String str = null; 
		
		
		double max_prediction = Double.MIN_VALUE;
		double min_prediction = Double.MAX_VALUE;
		int test_idx=0;
		int count_out_of_bound_nonExist = 0;
		int count_out_of_bound_exist = 0;
		
		LinkedList<Double> record_predictions = new LinkedList<Double>();
		LinkedList<Boolean> exist_record = new LinkedList<Boolean>();
		
		while((str=reader.readLine())!=null)
		{
			String[] pair = str.split(",");
			int item_id = Integer.valueOf(pair[0]);
			int user_id = Integer.valueOf(pair[1]);
			
			int count_unknown_user = 0;
			int count_unknown_item = 0;
			int exist_user_item_record = 0;
			int non_exist_user_item_record = 0;
			
			// unknown users, user average rating as substitute
			double predict_rating = 0.0;
			if(!this.valid_user_id.contains(user_id))
			{
				predict_rating = average_rating_vector.get(item_id);
				count_unknown_user ++;
				exist_record.add(false);
				record_predictions.add(predict_rating);
			}
			
			else if (!this.valid_item_id.contains(item_id))
			{
				predict_rating = average_rating;
				count_unknown_item++;
				exist_record.add(false);
				record_predictions.add(predict_rating);
			}
			
			// valid users and valid item, using its knn to predict
			else
			{
				Vector p = P.viewRow(user_id);
				Vector q = Q.viewColumn(item_id);
				
				// normal one, but may out of range of the ordinary ranking value
				// predict_rating = p.dot(q);
				
				// TODO
				// applied to the one using logic function
				// mapping the value back to [1,K]
				predict_rating = predict_rating_logic(p,q) * (this.K-1.0) + 1.0;
				record_predictions.add(predict_rating);
			}
			
			// write to file
			String rating1 = String.format("%.2f", predict_rating);
			writer.write(rating1);
			writer.newLine();
			writer.flush();
			
//			System.out.println("test "+test_idx + "\t(" + item_id + "," + user_id + ")\t\t" + rating1);
		    test_idx++;

		}
		
		reader.close();
		writer.close();
		System.out.println("---finished prediction");
		System.out.println("count_out_of_bound_nonExist:" + count_out_of_bound_nonExist);
		System.out.println("count_out_of_bound_exist:" + count_out_of_bound_exist);
		System.out.println("max value of exist prediction:"+max_prediction);
		System.out.println("min value of exist prediction:"+min_prediction);
	    
	}
    
    public void write_feature_vectors(String userFeatureFilePath, String itemFeatureFilePath) throws IOException
    {
    	
    	
    }
	
}
