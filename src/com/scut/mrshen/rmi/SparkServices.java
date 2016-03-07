package com.scut.mrshen.rmi;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface SparkServices extends Remote{
	// Linear Regression
	public String getLinearRegResult(String filename,  String[] args) throws RemoteException;
	// Logistic Regression with LBFGS
	public String getLogisticRegResult(String filename,  String[] args) throws RemoteException;
	// SVM with SGD
	public String getSVMResult(String filename,  String[] args) throws RemoteException;
	// MovieLen with custom args
	public String getMovienlenWithArgsResult(String filename, String[] args) throws RemoteException;
	// MovieLen with the best RSEM
	public String getBestMovienlenResult(String filename) throws RemoteException;
}
