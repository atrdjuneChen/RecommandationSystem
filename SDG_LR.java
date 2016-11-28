package CollaborativeRanking;


import org.apache.mahout.math.SparseMatrix;

import java.util.Random;
import java.io.*;
import java.util.*;
import org.apache.mahout.math.*;
import org.apache.mahout.math.Vector;
import com.google.common.base.Preconditions;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import org.apache.mahout.math.function.Functions;

public class SDG_LR {
	
	// multi-classes logic regression parameters 
	int F; 	 		 	// feature dimension
	int C;  		 	// number of classes
	Matrix Theta;		// target parameters

	// gradient descent parameters
	double threshold;
	int max_iter_time;
	double lambda;   // regularization parameter
	double alpha;    // learning rate
	
	// batch parameters
	int batch_size;		// optimal batch size, updated in the function of build_training_batches
	Matrix X_validate;	// small validation set when training
	Vector y_validate;  // small validation set when training
	List<String> batches_X; 	// X batches for batch-SDG
	
	
	public SDG_LR (int F, int C, double threshold, int max_iter_time, double alpha, double lambda) throws IOException
	{
		this.F = F;  // feature dimension
		this.C = C;	 // classes number
		
		this.Theta = new DenseMatrix(F, C).assign(Functions.random());
		this.threshold = threshold;
		this.max_iter_time = max_iter_time;
		this.alpha = alpha;  // learning rate
		this.lambda=lambda;  // the regularization parameter
		this.batches_X = new LinkedList<String>();  // FilePath of the batches
	}
	
	
	/**
	 * build training batches of Matrix
	 * @throws InterruptedException 
	 * */
	public void build_training_batches (String trainingPath, int user_defined_batch_size, String batchDir) throws IOException, InterruptedException
	{
		System.out.println("building training batches");
		String validate_batchFilePath = batchDir+"//validate";
		if(new File(validate_batchFilePath).exists())
		{
			System.out.println("bathes existed");
			System.out.println("----> finished\n");
			return;
		}
		
		long training_num = 0;
		File f = new File(trainingPath);
		BufferedReader reader = new BufferedReader(new FileReader(f));
		String str = null;
		while((str=reader.readLine())!=null)
			training_num++;
		reader.close();
		
		reader = new BufferedReader(new FileReader(f));
		str = null;
		System.out.println("\tnumber of input training data: " + training_num);
		// calculate optimal batch size
		int validata_num = (int) Math.round(training_num*0.02);
		System.out.println("\tnumber of validation data : " + validata_num);
		long actual_training_num = training_num - validata_num;
		System.out.println("\tnumber of acutual training data: " + actual_training_num);
		int batch_size = (int) (actual_training_num / (actual_training_num / user_defined_batch_size));
		this.batch_size = batch_size;
		System.out.println("\tuser defined batch size: "+ user_defined_batch_size);
		System.out.println("\toptimal batch size: " + batch_size);
		// System.out.println("  [use " + batch_size +" as batch size in order to get more evenly splitted batches]");
		int k = (int) (actual_training_num / batch_size);   // number of batches
		System.out.println("\tnumber of batch : "+k+"\n");
		
		// take 5% of the training data as validation set during training
		this.X_validate=new SparseMatrix(validata_num, this.F);
		this.y_validate=new DenseVector(validata_num);
		int count = 0;
		File f_validate = new File(validate_batchFilePath);
		BufferedWriter writer = new BufferedWriter(new FileWriter(f_validate));
		while((str=reader.readLine())!=null && count < validata_num)
		{
			writer.write(str);
			writer.newLine();
			count++;
		}
		writer.close();

		// the first (k-1) batches have the same size
		// construct the first (k-1) batches for training
		int batch_count = 0;
		String batchFilePath = batchDir+"//"+String.valueOf(batch_count+1);
		File fbatch = new File(batchFilePath);
		writer = new BufferedWriter(new FileWriter(fbatch));
		count = 0;
		while((str=reader.readLine())!=null && batch_count < (k-1))
		{
			writer.write(str);
			writer.newLine();
			count++;
			if(count == batch_size)
			{
				writer.close();
				batch_count++;
				//System.out.println("finished building batch" + String.valueOf(batch_count));
				batchFilePath = batchDir+"//"+String.valueOf(batch_count+1);
				writer = new BufferedWriter(new FileWriter(new File(batchFilePath)));
				count = 0;
			}
		}
					
		// the last batch might have more training examples
		batchFilePath = batchDir+"//"+String.valueOf(batch_count+1);
		writer = new BufferedWriter(new FileWriter(new File(batchFilePath)));
		while((str=reader.readLine())!=null)
		{
			writer.write(str);
			writer.newLine();
		}
		reader.close();
		writer.close();
		System.out.println("----> finished\n");
	}
	
	public void load_one_batch(String FilePath, Matrix X, Vector y) throws IOException
	{
		File f=new File(FilePath);
		BufferedReader reader = new BufferedReader(new FileReader(f));
		String str = null;
		List<Integer> labels = new LinkedList<Integer>();
		List<Vector> samples = new LinkedList<Vector>();
		while((str=reader.readLine())!=null)
		{
			String[] temp = str.split(" ");
			int label = temp[0].equals("1") ? 2 : 1;
			labels.add(label);
			Vector v = new DenseVector(this.F);
			for(int i=1; i<temp.length; i++)
			{
				String[] pair = temp[i].split(":");
				int idx = Integer.valueOf(pair[0])-1;
				double value = Double.valueOf(pair[1]);
				v.set(idx, value);
			}
			samples.add(v);
		}
		for(int i=0; i<samples.size(); i++)
		{
			X.assignRow(i, samples.get(i));
			y.set(i, labels.get(i));
		}
	}
	
	/**
	 * Training the Logistic Regression model with SDG
	 * @param trainingResultOutputPath
	 * @param trainingResultOutputPath_temp
	 * @throws IOException
	 * @throws InterruptedException
	 */
	public void train (String batchesPath, String trainingResultOutputPath, String trainingResultOutputPath_temp) throws IOException, InterruptedException
	{
		System.out.println("training");
		// loading batches path
		File fbatches = new File(batchesPath);
		File[] batches = fbatches.listFiles((dir, name) -> !(name.equals(".DS_Store")||name.equals("validate")));
		int batches_num = batches.length;
		HashMap<String, Integer> batch_info = new HashMap<String, Integer>();
		for(File batch : batches)
		{
			BufferedReader reader = new BufferedReader(new FileReader(batch));
			int count = 0;
			while(reader.readLine()!=null)
				count ++;
			reader.close();
			batch_info.put(batch.getName(), count);
		}
		// load validate X
		File fvalidate = (fbatches.listFiles((dir, name) -> name.equals("validate")))[0];
		BufferedReader reader = new BufferedReader(new FileReader(fvalidate));
		int count = 0;
		while(reader.readLine()!=null)
			count ++;
		reader.close();
		Matrix X_validate = new DenseMatrix(count, this.F);
		Vector y_validate = new DenseVector(count);
		load_one_batch(fvalidate.getPath(), X_validate, y_validate);
		
		// number of batches which are for training
		int M = batches_num;
		Random generator = new Random();
		double change = Double.MAX_VALUE;
		int iter_time=0;
		double bestCase = 0.90;
		System.out.println(String.format("%-10s%-10s%-20s%-20s" , "iter", "batch", "change", "ACC"));
		while(change > this.threshold && iter_time<this.max_iter_time)
		{
			int n = generator.nextInt(M); // randomly choose a batch to calculate average gradient
			File f_batch = batches[n];
			int size = batch_info.get(f_batch.getName());
			Matrix X = new DenseMatrix(size, this.F);
			Vector y = new DenseVector(size);
			load_one_batch(f_batch.getPath(), X,y);
			// calculate the gradient
			Matrix gradient = gradient(X, y);
			this.Theta = this.Theta.plus(gradient.times(this.alpha));
			// the change of gradient
			change = Math.sqrt(gradient.assign(Functions.SQUARE).zSum());
			
			// TODO
			// validate using hard prediction
			Vector validate_y_hard_predict = hard_predict(X_validate);
			//double RMSE_validate_hard = Math.sqrt(validate_y_hard_predict.getDistanceSquared(this.y_validate))/this.y_validate.size();
			//System.out.println(validate_y.asFormatString());
			//System.out.println(validate_y_predict.asFormatString());
			//System.out.println("validation set RMSE (hard predict): " + RMSE_validate_hard);
			
			//Vector validate_y_soft_predict = soft_predict(this.X_validate);
			//double RMSE_validate_soft = RMSE(validate_y_soft_predict, this.y_validate);
			double Accuracy = Accuracy(validate_y_hard_predict, y_validate);
			
			if(Accuracy > bestCase)
			{
				bestCase = Math.max(Accuracy, bestCase);
				File f = new File(trainingResultOutputPath_temp);
				BufferedWriter writer = new BufferedWriter(new FileWriter(f));
				for(int i=0; i<this.Theta.rowSize(); i++)
				{
					writer.write(this.Theta.viewRow(i).asFormatString());
					writer.newLine();
				}
				writer.close();
			}
			//System.out.println("validation set RMSE (soft predict): " + RMSE_validate_soft);
			
			System.out.println(String.format("%-10s%-10s%-20s%-20s" , String.valueOf(iter_time), 
					f_batch.getName(), String.valueOf(change), String.valueOf(Accuracy)));
			iter_time++;
		}
		
		File f = new File(trainingResultOutputPath);
		BufferedWriter writer = new BufferedWriter(new FileWriter(f));
		for(int i=0; i<this.Theta.rowSize(); i++)
		{
			writer.write(this.Theta.viewRow(i).asFormatString());
			writer.newLine();
		}
		writer.close();
		
		/*
		Vector validate_y_hard_predict = hard_predict(validate_X);
		Vector validate_y_soft_predict = soft_predict(validate_X);
		
		 System.out.println("groudn Truth:\n" +validate_y.asFormatString());
		 System.out.println("hard: \n"+ validate_y_hard_predict.asFormatString());
		 System.out.println("soft: \n"+validate_y_soft_predict.asFormatString());
		 Vector diff = validate_y_soft_predict.minus(validate_y);
		 */
		 System.out.println("----> finished\n");
	}
	
	public double RMSE(Vector y1, Vector y2)
	{
		return Math.sqrt(y1.getDistanceSquared(y2))/y2.size();
	}
	
	public double Accuracy(Vector y1, Vector y2)
	{
		Vector diff = y1.minus(y2);
		int error = diff.getNumNonZeroElements();
		return (y1.size()-error) * 1.0 / y1.size();
	}

	// hard prediction
	public Vector hard_predict (Matrix X)
	{
		Matrix Y_predict = X.times(this.Theta);
		Vector y_predict = new DenseVector(Y_predict.rowSize());
		for(int i=0; i<Y_predict.rowSize(); i++)
		{
			int predictClass = 1+Y_predict.viewRow(i).maxValueIndex();     // y ranking from 1 to 2, so here need to +1
			y_predict.set(i, predictClass);
		}
		return y_predict;
	}
	
	// soft prediction
	public Vector soft_predict (Matrix X)
	{
		Matrix A = X.times(this.Theta);
		
		// sum (axis=1)
		Vector B = (A.assign(Functions.EXP).times(new DenseVector(this.C).assign(1)));
		B = B.assign(Functions.INV);
		
		// broadcast
		Matrix P = new DiagonalMatrix(B).times(A);
		
		Vector temp = new DenseVector(this.C);
		for(int n=0; n<temp.size(); n++)
			temp.set(n, n+1);
		
		Vector y_predict = P.times(temp);
		return y_predict;
	}
	
	
	// calculate the average gradient with one or more samples
	public Matrix gradient(Matrix X_sample, Vector y_sample) throws InterruptedException
	{
		Matrix X_sample_T = X_sample.transpose();
		
		//  Y_ic=1 if y_i=c, otherwise Y_ic=0
		Matrix Y = new SparseMatrix(X_sample.rowSize(), this.C);
		for(int n=0; n<y_sample.size(); n++)
			Y.setQuick(n, (int) y_sample.get(n)-1, 1);   ////// y rating from 1 to K, here we need to -1
		
		Matrix gradient = new SparseMatrix(this.F, this.C);
		Matrix A = X_sample.times(this.Theta);
		
		// sum (axis=1)
		Vector B = (A.assign(Functions.EXP).times(new DenseVector(this.C).assign(1)));
		B = B.assign(Functions.INV);
		
		// broadcast
		Matrix C = new DiagonalMatrix(B).times(A);

		// Cij = Yic - Cij
		C = Y.minus(C);
		
		gradient = (X_sample_T.times(C)).minus(this.Theta.times(this.lambda));
		return gradient;
	}
	
	/**
	 * Classify
	 * @param testingPath
	 * @param outputFilePath
	 * @throws IOException
	 */
	void classify (String testingPath, String outputFilePath) throws IOException
	{
		System.out.println("classifying testing data");
		File f0 = new File(testingPath);
		BufferedReader reader = new BufferedReader(new FileReader(f0));
		String str = null;
		int test_num = 0;
		while((str=reader.readLine())!=null)
			test_num++;
		reader.close();
		
		// pack testing matrix X
		Matrix X_test = new SparseMatrix(test_num, this.F);
				
		reader = new BufferedReader(new FileReader(f0));
		str = null;
		
		int row = 0;
		while((str=reader.readLine())!=null)
		{
			Vector v = new RandomAccessSparseVector(this.F);
			String[] pairs = str.split(" ");
			for(int i=1; i<pairs.length; i++)
			{
				String[] p = pairs[i].split(":");
				int idx = Integer.valueOf(p[0])-1; 
				double value = Double.valueOf(p[1]);
				v.set(idx, value);
			}
			X_test.assignRow(row, v);
			row++;
		}
		
		File f = new File(outputFilePath);
		BufferedWriter writer = new BufferedWriter(new FileWriter(f));
		writer.write("labels 1 -1");
		writer.newLine();
		// validate using hard prediction
		Vector validate_y_hard_predict = hard_predict(X_test);
		Vector validate_y_soft_predict = soft_predict(X_test);
		for(int i=0; i<test_num; i++)
		{
			String label = (validate_y_hard_predict.get(i)==1) ? "-1" : "1";
			double prob_1 = 0.0;
			if(label.equals("1"))
				prob_1 = validate_y_soft_predict.get(i);
			else
				prob_1 = 1.0-validate_y_soft_predict.get(i);
			
			String str_write = label+ " " + prob_1 + " " + (1.0-prob_1);
			writer.write(str_write);
			writer.newLine();
		}
		writer.close();
		reader.close();
		System.out.println("----> finished\n");
	}
	
	
	/**
	 * Load LRModel
	 * @param preTrainedTheta
	 * @throws IOException
	 * input format : {0:-2.2782039469388016,1:3.302334953008314}
	 */
	public void loadModel(String preTrainedTheta) throws IOException
	{
		File f = new File(preTrainedTheta);
		BufferedReader reader = new BufferedReader(new FileReader(f));
		String str = null;
		int row = 0;
		while((str=reader.readLine())!=null)
		{
			String[] pair = str.substring(1, str.length()-1).split(",");
			Vector v = new DenseVector(this.C);
			for(int i=0; i<pair.length; i++)
			{
				String[] temp = pair[i].split(":");
				int idx = Integer.valueOf(temp[0]);
				double value = Double.valueOf(temp[1]);
				v.set(idx, value);
			}
			this.Theta.assignRow(row, v);
			row++;
		}
		reader.close();
	}
}
