package com.scut.mrshen.Models

import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.log4j.Level
import org.apache.log4j.Logger
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.SparkConf
import com.scut.mrshen.Config.SparkConfig

class MovieLensDemo {
/**
 * @author mrshen
 */
    def run(filename:String, args: Array[String]):String={
        Logger.getLogger(SparkConfig.APACHE_SPARK).setLevel(Level.WARN)
        Logger.getLogger(SparkConfig.JETTY_SERVER).setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setJars(Array(SparkConfig.RUNNABLE_JAR))
            .set(SparkConfig.EXECUTOR_MEMORY_KEY, SparkConfig.EXECUTOR_MEMORY_VAL)
            .set(SparkConfig.DRIVER_MEMORY_KEY, SparkConfig.DRIVER_MEMORY_VAL)
        val sc = new SparkContext(SparkConfig.MASTER_URL, "MovieLens with Custom Args", conf)
        
        var ratings:RDD[(Long, Rating)] = null
        if(filename.contains("1m")) {
            ratings = sc.textFile(SparkConfig.HDFS_ROOT_PATH + filename + "/ratings.dat").map { line =>
                val fields = line.split("::")
                (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
                }
        } else {
            ratings = sc.textFile(SparkConfig.HDFS_ROOT_PATH + filename + "/ratings.csv").map { line =>
                val fields = line.split(",")
                (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
                }
        }
        // TODO
        val numRatings = ratings.count()
        val numUser      = ratings.map(_._2.user).distinct().count()
        val numMovies    = ratings.map(_._2.product).distinct().count()
        /**/println("Got " + numRatings + " ratings from " + numUser + " users on " + numMovies + " movies.")
        
        val numPartitions = 20
        val training = ratings.filter(x => x._1 < 7)
                              .values
                              .repartition(numPartitions)
                              .persist()
        val test = ratings.filter(x => x._1 >= 7)
                          .values
                          .persist()
        /**/println("Training: " + training.count() + ", test: " + test.count())
        
        val rank = args(0).toInt
        val numIter = args(1).toInt
        val lambda = args(2).toDouble
        
        val model = ALS.train(training, rank, numIter, lambda)
        val RMSE = computeRMSE(model, test)
        println("RMSE (test) = " + RMSE + " for the model trained with rank = " 
                + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
        
        sc.stop()
        return new String("RMSE = " + RMSE + " for the model trained with rank = " 
                + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".");
    }
    
    
  


    def runAll(filename: String):String={
        Logger.getLogger(SparkConfig.APACHE_SPARK).setLevel(Level.WARN)
        Logger.getLogger(SparkConfig.JETTY_SERVER).setLevel(Level.OFF)
        
        val conf = new SparkConf()
            .setJars(Array(SparkConfig.RUNNABLE_JAR))
            .set(SparkConfig.EXECUTOR_MEMORY_KEY, SparkConfig.EXECUTOR_MEMORY_VAL)
            .set(SparkConfig.DRIVER_MEMORY_KEY, SparkConfig.DRIVER_MEMORY_VAL)
        val sc = new SparkContext(SparkConfig.MASTER_URL, "MovieLens", conf)
        
        var ratings:RDD[(Long, Rating)] = null
        if(filename.contains("1m")) {
            ratings = sc.textFile(SparkConfig.HDFS_ROOT_PATH + filename + "/ratings.dat").map { line =>
                val fields = line.split("::")
                (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
                }
        } else {
            ratings = sc.textFile(SparkConfig.HDFS_ROOT_PATH + filename + "/ratings.csv").map { line =>
                val fields = line.split(",")
                (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
                }
        }
        
        // TODO
        val numRatings = ratings.count()
        val numUser      = ratings.map(_._2.user).distinct().count()
        val numMovies    = ratings.map(_._2.product).distinct().count()
        println("Got " + numRatings + " ratings from " + numUser + " users on " + numMovies + " movies.")
        
        val numPartitions = 20
        val training = ratings.filter(x => x._1 < 6)
                                                    .values
                                                    .repartition(numPartitions)
                                                    .persist()
        val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
                                                        .values
                                                        .repartition(numPartitions)
                                                        .persist()
        val test = ratings.filter(x => x._1 > 8)
                                            .values
                                            .persist()
        println("Training: " + training.count() + ", validation: " + validation.count() + ", test: " + test.count())
        
        val ranks = List(8, 10, 12)
        val lambdas = List(0.1, 1)
        val numIters = List(10, 15, 20)
        var bestModel: Option[MatrixFactorizationModel] = None
        var bestValidationRmse = Double.MaxValue
        var bestRank = 0
        var bestLambda = -1.0
        var bestNumIter = -1
        for(rank <- ranks; numIter <- numIters; lambda <- lambdas) {
            val model = ALS.train(training, rank, numIter, lambda)
            val validationsRmse = computeRMSE(model, validation)
            println("RMSE (validation) = " + validationsRmse + " for the model trained with rank = " 
                    + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
            if(validationsRmse < bestValidationRmse) {
                bestModel = Some(model)
                bestValidationRmse = validationsRmse
                bestRank = rank
                bestLambda = lambda
                bestNumIter = numIter
            }
        }
        
        val testRmse = computeRMSE(bestModel.get, test)
        println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
                + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")
        
        sc.stop()
        return new String("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
                + ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".");
  }
    /** Compute RMSE (Root Mean Squared Error). */
    def computeRMSE(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {

        val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
        val predictionsAndRatings = predictions.map{ x =>
          ((x.user, x.product), x.rating)
        }.join(data.map(x => ((x.user, x.product), x.rating))).values
        math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
        }
}