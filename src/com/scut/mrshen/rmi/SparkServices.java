package com.scut.mrshen.rmi;

import java.rmi.Remote;
import java.rmi.RemoteException;

public interface SparkServices extends Remote{
	public String getResult(String dataPath, String masterUrl, String[] args) throws RemoteException;
}
