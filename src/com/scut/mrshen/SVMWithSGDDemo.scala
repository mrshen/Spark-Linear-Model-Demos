package com.scut.mrshen

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils

object SVMWithSGDDemo {
    val master = "spark://slaver2:7077"
    val dataSrc= "hdfs://slaver2:9000/tmp/sample_libsvm_data.txt"
    
    def main(args: Array[String]) {
        
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(master)
            .setJars(Array("/usr/local/spark/svmWithSGD.jar"))
            .set("spark.executor.memory", "6G")
            .set("spark.driver.memory", "6G")
            .setAppName("SVM With SGD")
            
        val sc = new SparkContext(conf)
        val data = MLUtils.loadLibSVMFile(sc, dataSrc)
        
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
        println("Area under ROC = " + auROC)
        
        sc.stop()
    }
}