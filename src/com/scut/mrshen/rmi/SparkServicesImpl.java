package com.scut.mrshen.rmi;

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;

import com.scut.mrshen.Models.LinearRegressionDemo;
import com.scut.mrshen.Models.LogisticRegressionDemo;
import com.scut.mrshen.Models.MovieLensDemo;
import com.scut.mrshen.Models.SVMDemo;

public class SparkServicesImpl extends UnicastRemoteObject implements SparkServices{

	private static final long serialVersionUID = -4393594638200648888L;

	public SparkServicesImpl() throws RemoteException {
		super();
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getLinearRegResult(String filename, String[] args)
			throws RemoteException {
		// TODO Auto-generated method stub
		return new LinearRegressionDemo().run(filename, args);
	}

	@Override
	public String getMovienlenWithArgsResult(String filename, String[] args)
			throws RemoteException {
		// TODO Auto-generated method stub
		return new MovieLensDemo().run(filename, args);
	}

	@Override
	public String getBestMovienlenResult(String filename)
			throws RemoteException {
		// TODO Auto-generated method stub
		return new MovieLensDemo().runAll(filename);
	}

	@Override
	public String getLogisticRegResult(String filename, String[] args)
			throws RemoteException {
		// TODO Auto-generated method stub
		return new LogisticRegressionDemo().run(filename, args);
	}

	@Override
	public String getSVMResult(String filename, String[] args)
			throws RemoteException {
		// TODO Auto-generated method stub
		return new SVMDemo().run(filename, args);
	}

}
