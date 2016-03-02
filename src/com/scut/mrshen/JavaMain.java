package com.scut.mrshen;

import com.scut.mrshen.Models.LinearRegressionDemo;
//import com.scut.mrshen.Models.LogisticRegressionDemo;
//import com.scut.mrshen.Models.SVMDemo;

public class JavaMain {
	private static final String masterUrl = "spark://slaver2:7077";
//	private static final String dataPath = "hdfs://slaver2:9000/tmp/sample_libsvm_data.txt";
	private static final String dataPath = "hdfs://slaver2:9000/tmp/lpsa.data";
	
	public static void main(String[] args) {
		// test LogisticRegression Demo
//		double res = new LogisticRegressionDemo().run(dataPath, masterUrl, args);
//		System.out.println("LR precision = " + res);
		
		// test SVM Demo
//		double res = new SVMDemo().run(dataPath, masterUrl, args);
//		System.out.println("Area of ROC = " + res);
		
		// test Linear Regression Demo
		double res = new LinearRegressionDemo().run(dataPath, masterUrl, args);
		System.out.println("MSE of LR = " + res);
	}
}
