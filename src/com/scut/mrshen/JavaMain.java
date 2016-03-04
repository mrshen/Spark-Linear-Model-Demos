package com.scut.mrshen;

import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;

import com.scut.mrshen.rmi.SparkServices;
import com.scut.mrshen.rmi.SparkServicesImpl;
//import com.scut.mrshen.Models.LogisticRegressionDemo;
//import com.scut.mrshen.Models.SVMDemo;

public class JavaMain {
//	private static final String masterUrl = "spark://slaver2:7077";
//	private static final String dataPath = "hdfs://slaver2:9000/tmp/sample_libsvm_data.txt";
//	private static final String dataPath = "hdfs://slaver2:9000/tmp/lpsa.data";
	
	public static void main(String[] args) {
		// test LogisticRegression Demo
//		double res = new LogisticRegressionDemo().run(dataPath, masterUrl, args);
//		System.out.println("LR precision = " + res);
		
		// test SVM Demo
//		double res = new SVMDemo().run(dataPath, masterUrl, args);
//		System.out.println("Area of ROC = " + res);
		
		// test Linear Regression Demo
//		double res = new LinearRegressionDemo().run(dataPath, masterUrl, args);
//		System.out.println("MSE of LR = " + res);
		
		try {
			SparkServices services = new SparkServicesImpl();
			LocateRegistry.createRegistry(8888);
			Naming.rebind("rmi://slaver2:8888/SparkServices", services);
			System.out.println("rmi service ready..");
		} catch (RemoteException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (MalformedURLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
