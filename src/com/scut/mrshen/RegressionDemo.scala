package com.scut.mrshen

import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionWithSGD

object RegressionDemo {
    val master = "spark://slaver2:7077"
    val dataSrc= "hdfs://slaver2:9000/tmp/lpsa.data"
    
    def main(args: Array[String]) {
        
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
        Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setMaster(master)
            .setJars(Array("/usr/local/spark/regression.jar"))
            .set("spark.executor.memory", "6G")
            .set("spark.driver.memory", "6G")
            .setAppName("regression")
            
        val sc = new SparkContext(conf)
        val data = sc.textFile(dataSrc)
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
        println("training Mean Squared Error = " + MSE)    
        sc.stop()
    }
}