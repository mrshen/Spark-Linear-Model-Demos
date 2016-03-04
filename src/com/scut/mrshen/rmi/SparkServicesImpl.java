package com.scut.mrshen.rmi;

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

import com.scut.mrshen.Models.LinearRegressionDemo;

public class SparkServicesImpl extends UnicastRemoteObject implements SparkServices{

	public SparkServicesImpl() throws RemoteException {
		super();
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getResult(String dataPath, String masterUrl, String[] args)
			throws RemoteException {
		// TODO Auto-generated method stub
		double res = new LinearRegressionDemo().run(dataPath, masterUrl, args);
		return new String("" + res);
	}

}
