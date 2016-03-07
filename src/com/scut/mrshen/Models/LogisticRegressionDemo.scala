package com.scut.mrshen.Models

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import breeze.macros.expand.args
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.log4j.Level
import org.apache.log4j.Logger
import breeze.macros.expand.args
import org.apache.spark.mllib.regression.LabeledPoint
import com.scut.mrshen.Config.SparkConfig

class LogisticRegressionDemo {
    def run(filename: String, args: Array[String]):String={
        Logger.getLogger(SparkConfig.APACHE_SPARK).setLevel(Level.WARN)
        Logger.getLogger(SparkConfig.JETTY_SERVER).setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(SparkConfig.MASTER_URL)
            .setJars(Array(SparkConfig.RUNNABLE_JAR))
            .set(SparkConfig.EXECUTOR_MEMORY_KEY, SparkConfig.EXECUTOR_MEMORY_VAL)
            .set(SparkConfig.DRIVER_MEMORY_KEY, SparkConfig.DRIVER_MEMORY_VAL)
            .setAppName("LogisticRegression with L-BFGS")
            
        val sc = new SparkContext(conf)
        val data = MLUtils.loadLibSVMFile(sc, SparkConfig.HDFS_ROOT_PATH + filename)
        
        val splits = data.randomSplit(Array(args(0).toDouble, args(1).toDouble), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        val model = new LogisticRegressionWithLBFGS()
            .setNumClasses(args(2).toInt)
            .run(training)
        
        val predAndLabels = test.map{ case LabeledPoint(label, features) =>
            val pred = model.predict(features)
            (pred, label)
        }
        println("========================show (prediction, label)============================")
        predAndLabels.collect().foreach(println)
        println("=============================end show=======================================")
        val metrics = new MulticlassMetrics(predAndLabels)
        val precision = metrics.precision
//        println("Precision = " + precision)
        
        sc.stop()
        return new String("LogisticRegression Precision = " + precision)
    }
}