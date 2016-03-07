package com.scut.mrshen.Models

import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.SparkContext
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import com.scut.mrshen.Config.SparkConfig

class LinearRegressionDemo {
    def run(filename:String, args:Array[String]):String={
        Logger.getLogger(SparkConfig.APACHE_SPARK).setLevel(Level.WARN)
        Logger.getLogger(SparkConfig.JETTY_SERVER).setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(SparkConfig.MASTER_URL)
            .setJars(Array(SparkConfig.RUNNABLE_JAR))
            .set(SparkConfig.EXECUTOR_MEMORY_KEY, SparkConfig.EXECUTOR_MEMORY_VAL)
            .set(SparkConfig.DRIVER_MEMORY_KEY, SparkConfig.DRIVER_MEMORY_VAL)
            .setAppName("linear regression")
            
        val sc = new SparkContext(conf)
        val data = sc.textFile(SparkConfig.HDFS_ROOT_PATH + filename)
        val parsedData = data.map { line =>
            val parts = line.split(',')
            LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
        }.cache()
        
        val model = LinearRegressionWithSGD.train(parsedData, args(0).toInt)
        
        val valsAndPreds = parsedData.map { point =>
            val pred = model.predict(point.features)
            (point.label, pred)
        }
        println("========================show (label, prediction)============================")
        valsAndPreds.collect().foreach(println)
        println("=============================end show=======================================")
        val MSE = valsAndPreds.map {case(v, p) => math.pow((v-p), 2)}.mean()
//        println("training Mean Squared Error = " + MSE)
        sc.stop()
        return new String("training Mean Squared Error = " + MSE + " with interation of " + args(0));
    }
}