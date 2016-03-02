package com.scut.mrshen.Models

import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.SparkContext
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

class LinearRegressionDemo {
    def run(dataPath:String, masterUrl:String, args:Array[String]):Double={
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(masterUrl)
            .setJars(Array("/usr/local/spark/SparkMlModel.jar"))
            .set("spark.executor.memory", "6G")
            .set("spark.driver.memory", "6G")
            .setAppName("regression")
            
        val sc = new SparkContext(conf)
        val data = sc.textFile(dataPath)
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
        return MSE
    }
}