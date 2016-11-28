package CollaborativeRanking;

import java.io.*;
import java.util.*;

public class SVM {

	String svmRankLearnPath = "";
	String svmRankClassifyPath;
	String trainingFeatureVectorsFile;
	String svmRankModelFile;
	
	
	
	public SVM (String svmRankLearnPath, String svmRankClassifyPath, 
			String trainingFeatureVectorsFile, String svmRankModelFile)
	{
		this.svmRankLearnPath = svmRankLearnPath;
		this.svmRankClassifyPath = svmRankClassifyPath;
		this.trainingFeatureVectorsFile = trainingFeatureVectorsFile;
		this.svmRankModelFile = svmRankModelFile;
	}
	
	
	
	// train svm
	// execPath is the location of the svm_rank_learn utility, 
    // which is specified by letor:svmRankLearnPath in the parameter file.
    // c is the value of the letor:c parameter.
	public void train() throws Exception
	{
		System.out.println("training svm");
		
		String execPath = this.svmRankLearnPath;
		String trainingFeatureFile = this.trainingFeatureVectorsFile;
		String modelOutputFile = this.svmRankModelFile;
		
		Process cmdProc = Runtime.getRuntime().exec(
		        new String[] { execPath, "-s", "0", trainingFeatureFile,
		            modelOutputFile });
		
		// The stdout/stderr consuming code MUST be included.
	    // It prevents the OS from running out of output buffer space and stalling.

	    // consume stdout and print it out for debugging purposes
	    BufferedReader stdoutReader = new BufferedReader(
	        new InputStreamReader(cmdProc.getInputStream()));
	    String line;
	    while ((line = stdoutReader.readLine()) != null) {
	      System.out.println(line);
	    }
	    // consume stderr and print it for debugging purposes
	    BufferedReader stderrReader = new BufferedReader(
	        new InputStreamReader(cmdProc.getErrorStream()));
	    while ((line = stderrReader.readLine()) != null) {
	      System.out.println(line);
	    }
	    
	    // get the return value from the executable. 0 means success, non-zero 
	    // indicates a problem
	    int retValue = cmdProc.waitFor();
	    if (retValue != 0) {
	      throw new Exception("SVM Rank crashed.");
	    }
	    System.out.println("----> finished\n");
	}
	
	
	/**
	 * use probability_estimates to classify
	 * @param testingFeatureVectorsFile
	 * @param testOutputFile
	 * @throws Exception
	 */
	public void classify(String testingFeatureVectorsFile, String testOutputFile) throws Exception
	{
		System.out.println("classifying testing data");
		String execPath = this.svmRankClassifyPath;
		String svmRankModelFile = this.svmRankModelFile;
		
		System.out.println(testOutputFile);
		
		Process cmdProc = Runtime.getRuntime().exec(
				new String[] { execPath, "-b", "1", testingFeatureVectorsFile, svmRankModelFile, testOutputFile});
		
		// consume stdout and print it out for debugging purposes
	    BufferedReader stdoutReader = new BufferedReader(
	        new InputStreamReader(cmdProc.getInputStream()));
	    String line;
	    while ((line = stdoutReader.readLine()) != null) {
	      System.out.println(line);
	    }
	    // consume stderr and print it for debugging purposes
	    BufferedReader stderrReader = new BufferedReader(
	        new InputStreamReader(cmdProc.getErrorStream()));
	    while ((line = stderrReader.readLine()) != null) {
	      System.out.println(line);
	    }

	    // get the return value from the executable. 0 means success, non-zero 
	    // indicates a problem
	    int retValue = cmdProc.waitFor();
	    if (retValue != 0) {
	      throw new Exception("SVM Rank crashed.");
	    }
	    System.out.println("----> finished\n");
	}

}
