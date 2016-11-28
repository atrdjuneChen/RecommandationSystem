package CollaborativeRanking;

import java.util.*;
import java.io.*;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import com.google.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;

/**
 * Driver of Collaborative filtering
 * 
 * Note: imputation is implemented here
 * 
 * */
public class Driver {

	private String training_path;
	private String dev_file_path;
	private String testing_file_path;
	private String outputfile_path;
	private String cf_method;
	HashSet<Integer> valid_user_id = new HashSet<Integer>();
	HashSet<Integer> valid_item_id = new HashSet<Integer>();
	private Matrix R;				  // storing the rating (after preprocess) 
	private Matrix Rating_records;    // if user i has rated item j, Rating_records(i,j)==1
	private Vector average_rating_vector;
	private int Ranking_K=5;      // ranking is from 1 to 5
	private int feature_num=50;
	HashMap<Integer,HashMap<Integer,Integer>> map;  // HashMap form of Rating Matrix R
	
	// hyperameter for PMF
	private double learning_rate = 0.005;
	private double beta1 = 0.0002;
	private double beta2 = 0.0002;
	private int maxStepNum = 100;
	private double threshold = 0.01;
	private boolean choose_test = false;   // false: development,  true: test
	HashMap<Integer, HashMap<Integer, Double>> mini_validation_set = new HashMap<Integer, HashMap<Integer, Double>>();
	
	// parameters for Letor
	String prediction_method="";
	// (svm)
	String svmRankLearnPath;
	String svmRankClassifyPath;
	// (lr)
	double lr_threshold;
	int lr_max_iter_time;
	double lr_learning_rate;
	double lr_lambda;
	int lr_sdg_batch_size;

	// loading the parameters from the arguments
	public Driver(String args) throws IOException
	{
		Map<String, String> parameters = readParameterFile (args);
		this.cf_method = parameters.get("cf_method");   // MemoryBased, ModelBased, PCCBased, PMFBased
		this.training_path = parameters.get("training_file_path");
		this.dev_file_path = parameters.get("dev_file_path");
		this.testing_file_path = parameters.get("testing_file_path");
		this.outputfile_path = parameters.get("outputfile_path");
		
		// parameters of PMF
		this.learning_rate=Double.valueOf(parameters.get("learning_rate"));
		this.beta1=Double.valueOf(parameters.get("beta1"));
		this.beta2=Double.valueOf(parameters.get("beta2"));
		this.maxStepNum=Integer.valueOf(parameters.get("maxStepNum"));
		this.threshold=Double.valueOf(parameters.get("threshold"));
		this.choose_test=Boolean.valueOf(parameters.get("choose_test"));
		this.feature_num = Integer.valueOf(parameters.get("feature_num"));
		
		// paramters of letor
		// svm
		this.prediction_method = parameters.get("letor:prediction_method");
		this.svmRankLearnPath = parameters.get("letor:svmRankLearnPath");
		this.svmRankClassifyPath = parameters.get("letor:svmRankClassifyPath");
		// lr
		this.lr_threshold = Double.valueOf(parameters.get("lr:threshold"));
		this.lr_max_iter_time = Integer.valueOf(parameters.get("lr:max_iter_time"));
		this.lr_learning_rate =Double.valueOf(parameters.get("lr:learning_rate"));
		this.lr_lambda =Double.valueOf(parameters.get("lr:lambda"));
	    this.lr_sdg_batch_size = Integer.valueOf(parameters.get("lr:sdg_batch_size"));
		
	}
	
	
	
	
	/**
	 * load training data and do preprocessing
	 * for PMFBased method, we scaled the rating from [1,K] to [0,1]
	 * */
	public void load_training_data () throws IOException
	{
		// load training data
		FileInputStream is=new FileInputStream(this.training_path);
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		String str = null; 
		this.map = new HashMap<Integer,HashMap<Integer,Integer>>();
		
		// record all rating of each item
		HashMap<Integer, List<Integer>> item_rating = new HashMap<Integer, List<Integer>>();
		
        int max_user_id = Integer.MIN_VALUE;
        int max_item_id = Integer.MIN_VALUE;
        
		while ((str = reader.readLine())!=null)
        {
            String[]temp = str.split(",");
            int item_id = Integer.valueOf(temp[0]);
            int user_id = Integer.valueOf(temp[1]);
            int rating = Integer.valueOf(temp[2]);
            this.valid_user_id.add(user_id);
            this.valid_item_id.add(item_id);
            if(user_id>max_user_id)
            	max_user_id = user_id;
            if(item_id>max_item_id)
        		max_item_id = item_id;
            
            if(!this.map.containsKey(user_id))
            {
            	HashMap<Integer,Integer> current_user = new HashMap<Integer, Integer>();  // (item, rating)
            	this.map.put(user_id, current_user);
            }
            this.map.get(user_id).put(item_id, rating);
            
            if(!item_rating.containsKey(item_id))
            	item_rating.put(item_id, new LinkedList<Integer>());
            item_rating.get(item_id).add(rating);
        }
		
		int max_count = 0;
		for(Integer userId : this.map.keySet())
		{
			int records = map.get(userId).size();
			max_count = Math.max(max_count, records);
		}
		System.out.println("user at most rating "+max_count+" items in this training data");
		int[] historgram = new int[max_count+1];
		
		for(Integer userId : this.map.keySet())
		{
			int records = map.get(userId).size();
			historgram[records]++;
		}
		
		
		System.out.println("#valid user id : "+valid_user_id.size());
		System.out.println("maximum user id : " + max_user_id);
		System.out.println("#valid item id : "+valid_item_id.size());
		System.out.println("maximum item id : " + max_item_id);
		if(this.valid_item_id.contains(0))
			System.out.println("valid item id starts from 0");
		if(this.valid_user_id.contains(0))
			System.out.println("valid user id starts from 0");
		
		//  a dense vector to store the average rating of each item
		this.average_rating_vector = new DenseVector (max_item_id+1);
		for(Integer item_id : item_rating.keySet())
		{
			int sum = 0;
			for(Integer rating : item_rating.get(item_id))
				sum += rating;
			
			// TODO if I use ceil, some case will get a prediction of 6.0, for the cold start case.
			double average = Math.floor ( 1.0 * sum/item_rating.get(item_id).size() );
			this.average_rating_vector.setQuick(item_id, average);
		}
		
		// construct a user-item rating_values sparse matrix and a user-item recording sparse matrix
        this.R = new SparseMatrix(max_user_id+1, max_item_id+1); 
        this.Rating_records = new SparseMatrix(max_user_id+1, max_item_id+1); 
        for(Integer user_id : this.map.keySet())
        {
        	Vector v_record = new RandomAccessSparseVector(max_item_id + 1);
        	Vector v = new RandomAccessSparseVector(max_item_id + 1);
        	for(Integer item_id : this.map.get(user_id).keySet())
        	{	
        		v_record.setQuick(item_id, 1);
	        	if(!this.cf_method.equals("PMFBased"))
	        	{
	        		v.setQuick(item_id, this.map.get(user_id).get(item_id)-3);   // imputation (substract 3 from non-empty cells)
	        		// in case when a user only rated one item with 3, after imputation, it will become a all-zero vector
	            	// so use the average_rating_vectors as a substitute
	            	if(v.getNumNonZeroElements()==0)
	            		v = this.average_rating_vector.clone();
	        	}
	        	// for PMFBased method, we map the rating between [0,1] , 1 will become 0, and K will become 1
	        	else
	        	{
	        		// TODO
	        	    double scaled_value = (this.map.get(user_id).get(item_id) - 1.0) / (this.Ranking_K - 1.0);
	        		v.setQuick(item_id, scaled_value);
	        	}
        	}
        	
        	this.R.assignRow(user_id, v);   // so we place the vector according to the user_idx_in_matrix rather than the user_idx
        	this.Rating_records.assignRow(user_id, v_record);
        }
        
        // TODO needs second check
    	// construct mini validation set which can be used during training, 500 users, 3 items per user. ==> totally about 1500 samples
    	HashMap<Integer, HashMap<Integer, Double>> MiniValidationSet = new HashMap<Integer, HashMap<Integer, Double>>();
    	Random generator = new Random();
    	Object[] valid_user_ids = this.valid_user_id.toArray();
    	while(MiniValidationSet.size() < 500)
		{
    		Object randomUserId = valid_user_ids[generator.nextInt(valid_user_ids.length)];
    		if(this.map.get((int)randomUserId).size() < 3)
    			continue;
    		if(!MiniValidationSet.containsKey(randomUserId))
    		{
    			Object[] valid_item_id = this.map.get((int)randomUserId).keySet().toArray();
    			HashMap<Integer, Double> test_item = new HashMap<Integer, Double>();
    			while(test_item.size()<3)
    			{
    				Object randomItemId = valid_item_id[generator.nextInt(valid_item_id.length)];
    				if(!test_item.containsKey((int)randomItemId))
    				{
    					int rating = this.map.get((int)randomUserId).get((int)randomItemId);
    	    			double scaled_value = (rating - 1.0) / (this.Ranking_K - 1.0);
    	    			test_item.put((int)randomItemId, scaled_value);
    				}
    			}
    			MiniValidationSet.put((int)randomUserId, test_item);
    		}
		}
    	this.mini_validation_set = MiniValidationSet;
        reader.close();
        
	}
	
	
	
	
	
	
	public static void main(String[] args) throws Exception
	{
		Driver driver = new Driver(args[0]);
		
		String userFeatureFilePath = driver.outputfile_path+"//train//UserFeature"+driver.feature_num;
		String itemFeatureFilePath = driver.outputfile_path+"//train//ItemFeature"+driver.feature_num;

		driver.load_training_data();
		driver.print_configurations();
		
		if(!(new File(userFeatureFilePath)).exists() || !(new File(itemFeatureFilePath)).exists())
		{
			long startTime=0;
			long endTime=0;
			
			int feature_num = driver.feature_num;
			
			switch (driver.cf_method) {	        	
		        case "PMFBased":
	        		int number_latenfeature = feature_num;
	        		double learning_rate = driver.learning_rate;
	        		double beta1 = driver.beta1;
	        		double beta2 = driver.beta2;
	        		int maxStepNum = driver.maxStepNum;
	        		double threshold = driver.threshold;

	        		startTime=System.currentTimeMillis(); 
		        	PMFBasedCF pmf = new PMFBasedCF(driver.R, driver.Ranking_K, driver.Rating_records, driver.valid_user_id, driver.valid_item_id, driver.average_rating_vector,
		        			number_latenfeature, learning_rate, beta1, beta2, maxStepNum, threshold, driver.mini_validation_set, driver.map);
		        	pmf.solve();
		        	pmf.write_feature_vectors(userFeatureFilePath, itemFeatureFilePath);
		        	if(driver.choose_test==false)
		        		pmf.predict(driver.dev_file_path, driver.outputfile_path+"//dev");
		        	else
		        		pmf.predict(driver.testing_file_path, driver.outputfile_path+"//test");
		        	endTime=System.currentTimeMillis(); 
					System.out.println("run time for PMF: "+(double)(endTime-startTime)/1000);
			        break;
			        
		        default: 
		        	break;
			}
		}
		
		
		
		else
		{
			System.out.println("\n*********************************************");
			System.out.println("RANKING\n");
			// setting the file path
			Ranking ranking = new Ranking (driver.map, driver.valid_user_id, driver.valid_item_id);
			String trainingFeatureVectorsFile = driver.outputfile_path+"//train//PairWiseFeature"+driver.feature_num;
			String testDataInputFilePath = "";
			String testingFeatureVectorsFile= "";
			String classifyOutputFile="";
			String rankingOutputFile="";
			String rankingOutputFile_detailed="";
			if(driver.choose_test==true)
			{
				testDataInputFilePath = driver.testing_file_path;
				testingFeatureVectorsFile = driver.outputfile_path+"//test//PairWiseFeature"+driver.feature_num;
				classifyOutputFile = driver.outputfile_path+"//test//classify"+driver.feature_num;
				rankingOutputFile = driver.outputfile_path+"//test//ranking"+driver.feature_num;
				rankingOutputFile_detailed = driver.outputfile_path+"//test//ranking_detailed"+driver.feature_num;
			}
			else
			{
				testDataInputFilePath = driver.dev_file_path;
				testingFeatureVectorsFile = driver.outputfile_path+"//dev//PairWiseFeature"+driver.feature_num;
				classifyOutputFile = driver.outputfile_path+"//dev//classify"+driver.feature_num;
				rankingOutputFile = driver.outputfile_path+"//dev//ranking"+driver.feature_num;
				rankingOutputFile_detailed = driver.outputfile_path+"//dev//ranking_detailed"+driver.feature_num;
			}
			
			// build PairWiseFeature
			ranking.loadingFeature(userFeatureFilePath, itemFeatureFilePath);
			if(!(new File(trainingFeatureVectorsFile)).exists())
				ranking.buildingPairFeatureVectors_train(trainingFeatureVectorsFile);
			if(!(new File(testingFeatureVectorsFile)).exists())
				ranking.buildingPairFeatureVectors_test(testDataInputFilePath, testingFeatureVectorsFile);
			
			HashMap<Integer, List<Integer>> queries_per_user = new HashMap<Integer, List<Integer>>();
			List<String> item_user_pairs_order = new LinkedList<String>();
			driver.combine_queries(testDataInputFilePath, queries_per_user, item_user_pairs_order);
		
			if(driver.prediction_method.equals("svm"))
			{
				classifyOutputFile = classifyOutputFile+"_svm";
				rankingOutputFile = classifyOutputFile + "_svm";
				rankingOutputFile_detailed = rankingOutputFile_detailed + "_svm";
				String svmRankModelFile=driver.outputfile_path+"//train//SvmModel"+driver.feature_num;
				SVM svm = new SVM(driver.svmRankLearnPath, driver.svmRankClassifyPath, 
									trainingFeatureVectorsFile, svmRankModelFile);
				if(!(new File(svmRankModelFile)).exists())
					svm.train();
				if(!(new File(classifyOutputFile)).exists())
					svm.classify(testingFeatureVectorsFile, classifyOutputFile);
				ranking.rank(classifyOutputFile, rankingOutputFile, rankingOutputFile_detailed, 
								queries_per_user, item_user_pairs_order, "svm");
			}
			else if(driver.prediction_method.equals("lr"))
			{
				classifyOutputFile = classifyOutputFile + "_lr";
				rankingOutputFile = rankingOutputFile + "_lr";
				rankingOutputFile_detailed = rankingOutputFile_detailed + "_lr";
				String lrModelFile = driver.outputfile_path+"//train//LRModel"+driver.feature_num;
			
				String trainingResultOutputPath_temp = driver.outputfile_path+"//train//LRModel_temp"+driver.feature_num;
				SDG_LR sdg_lr = new SDG_LR (driver.feature_num, 2, driver.lr_threshold, driver.lr_max_iter_time, 
				    						driver.lr_learning_rate, driver.lr_lambda);
				if(!(new File(lrModelFile)).exists())
				{
				    String batchDir = driver.outputfile_path+"//train//batches";
				    sdg_lr.build_training_batches (trainingFeatureVectorsFile, driver.lr_sdg_batch_size, batchDir);
				    sdg_lr.train(batchDir, lrModelFile, trainingResultOutputPath_temp);
				}
				else 
					sdg_lr.loadModel(lrModelFile);
				sdg_lr.classify(testingFeatureVectorsFile, classifyOutputFile);
				
				ranking.rank(classifyOutputFile, rankingOutputFile, rankingOutputFile_detailed, 
								queries_per_user, item_user_pairs_order ,"lr");
			}
			else
				System.out.println("this program only support svm or logistic regression, type 'svm' or 'lr'");
			System.out.println("finished ranking");
			System.out.println("*********************************************\n");
		}
		
		
		
		

		
		
		System.out.println("\nFINISH ALL");
	}
	
	public void combine_queries(String testDataFilePath, HashMap<Integer, List<Integer>> queries_per_user,
			List<String> item_user_pairs_order) throws IOException
	{
		System.out.println("combined queries into their same users");
		File f = new File(testDataFilePath);
		BufferedReader reader = new BufferedReader(new FileReader(f));
		String str = null;
		while((str = reader.readLine())!=null)
		{
			item_user_pairs_order.add(str);
			String[] temp = str.split(",");
			int user=Integer.valueOf(temp[1]);
			int item=Integer.valueOf(temp[0]);
			if(!queries_per_user.containsKey(user))
				queries_per_user.put(user, new LinkedList<Integer>());
			queries_per_user.get(user).add(item);
		}
		reader.close();
		System.out.println("----> finished\n");
	}
	
	
	private static Map<String, String> readParameterFile (String parameterFileName) throws IOException {
	    Map<String, String> parameters = new HashMap<String, String>();
	    File parameterFile = new File (parameterFileName);
	    if (! parameterFile.canRead ()) 
	    {
	      throw new IllegalArgumentException
	        ("Can't read " + parameterFileName);
	    }
	    Scanner scan = new Scanner(parameterFile);
	    String line = null;
	    do {
	      line = scan.nextLine();
	      String[] pair = line.split ("=");
	      parameters.put(pair[0].trim(), pair[1].trim());
	    } while (scan.hasNext());
	    scan.close();
	    return parameters;
	}  
	
	/**
	 * print the configuration of parameters
	 * */
	private void print_configurations()
	{
		System.out.println("\n*********************************************");
		System.out.println("CONFIGURATIONS");
		System.out.println("training file path: "+ this.training_path);
		if(this.choose_test==false)
			System.out.println("development environment\ndata set for prediction: " + this.dev_file_path);
		else
			System.out.println("testing environment,  data set for prediction: " + this.testing_file_path);
		System.out.println("output file path: "+ this.outputfile_path);
		System.out.println("CF method: "+this.cf_method);
		if(this.cf_method.equals("PMFBased"))
		{
			System.out.println("feature_num: "+this.feature_num);
			System.out.println("learning_rate: "+this.learning_rate);
			System.out.println("beta1: "+this.beta1);
			System.out.println("beta2: "+this.beta2);
			System.out.println("maxStepNum: "+this.maxStepNum);
			System.out.println("threshold: "+this.threshold);
			System.out.println("prediction method: "+this.prediction_method);
		}
		System.out.println("*********************************************\n");
	}
	
	
}
