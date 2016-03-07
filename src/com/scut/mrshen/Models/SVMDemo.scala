package com.scut.mrshen.Models

import org.apache.spark.SparkConf
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.log4j.Level
import org.apache.log4j.Logger
import com.scut.mrshen.Config.SparkConfig

class SVMDemo {
    def run(filename: String, args: Array[String]): String={
        Logger.getLogger(SparkConfig.APACHE_SPARK).setLevel(Level.WARN)
        Logger.getLogger(SparkConfig.JETTY_SERVER).setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(SparkConfig.MASTER_URL)
            .setJars(Array(SparkConfig.RUNNABLE_JAR))
            .set(SparkConfig.EXECUTOR_MEMORY_KEY, SparkConfig.EXECUTOR_MEMORY_VAL)
            .set(SparkConfig.DRIVER_MEMORY_KEY, SparkConfig.DRIVER_MEMORY_VAL)
            .setAppName("SVM With SGD")
            
        val sc = new SparkContext(conf)
        val data = MLUtils.loadLibSVMFile(sc, SparkConfig.HDFS_ROOT_PATH + filename)
        
        val splits = data.randomSplit(Array(args(0).toDouble, args(1).toDouble), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)
        
        val model = SVMWithSGD.train(training, args(2).toInt)
        model.clearThreshold()
        
        val scoreAndLabels = test.map { point =>
            val score = model.predict(point.features)
            (score, point.label)
        }
        println("========================show (prediction, label)============================")
        scoreAndLabels.collect().foreach(println)
        println("=============================end show=======================================")
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()
//        println("Area under ROC = " + auROC)
        
        sc.stop()
        return new String("Area under ROC = " + auROC)
    }
}