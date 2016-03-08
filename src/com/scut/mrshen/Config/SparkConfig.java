package com.scut.mrshen.Config;

public class SparkConfig {
	// spark master url
	public static final String MASTER_URL = "spark://slaver2:7077";
	// name and path of runnable jar for spark
	public static final String RUNNABLE_JAR = "/usr/local/spark/SparkMlModel.jar";
	// spark executor memory key
	public static final String EXECUTOR_MEMORY_KEY = "spark.executor.memory";
	// spark executor memory val
	public static final String EXECUTOR_MEMORY_VAL = "6G";
	// spark driver memory key
	public static final String DRIVER_MEMORY_KEY = "spark.driver.memory";
	// spark driver memory key
	public static final String DRIVER_MEMORY_VAL = "6G";
	// test files root path
	public static final String HDFS_ROOT_PATH = "hdfs://slaver2:9000/tmp/";
	// apache spark name
	public static final String APACHE_SPARK = "org.apache.spark";
	// eclipse jetty server
	public static final String JETTY_SERVER = "org.eclipse.jetty.server";
}
