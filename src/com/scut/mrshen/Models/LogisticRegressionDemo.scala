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

class LogisticRegressionDemo {
    def run(dataPath: String, masterUrl: String, args: Array[String]):Double={
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(masterUrl)
            .setJars(Array("/usr/local/spark/SparkMlModel.jar"))
            .set("spark.executor.memory", "6G")
            .set("spark.driver.memory", "6G")
            .setAppName("LR with L-BFGS")
            
        val sc = new SparkContext(conf)
        val data = MLUtils.loadLibSVMFile(sc, dataPath)
        
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
        return precision
    }
}