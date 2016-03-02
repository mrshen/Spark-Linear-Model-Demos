package com.scut.mrshen

import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.SparkContext
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.rdd.RDD

/**
 * @author mrshen
 */
object MovieLen {
	def main(args: Array[String]) {
//		Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
//		Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
		
		val master = "spark://slaver2:7077"
		val dataSrc= "hdfs://slaver2:9000/user/hadoop/MovieLens/ml-1m/"
		
		val conf = new SparkConf()
			.setMaster(master)
            .setJars(Array("/usr/local/spark/movielens.jar"))
			.set("spark.executor.memory", "6G")
			.set("spark.driver.memory", "6G")
			.setAppName("MovieLens in Linux Eclipse")
		val sc = new SparkContext(conf)
		
		val ratings = sc.textFile(dataSrc + "ratings.dat").map { line =>
			val fields = line.split("::")
			(fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
			}
//        println(s"total lines: ${ratings.count()}")
		val movies  = sc.textFile(dataSrc + "movies.dat").map { line =>
			val fields = line.split("::")
			(fields(0).toInt, fields(1))
		}.collect.toMap
		
		println("current ok..")
		// TODO
		val numRatings = ratings.count()
		val numUser		 = ratings.map(_._2.user).distinct().count()
		val numMovies	 = ratings.map(_._2.product).distinct().count()
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
		
		val ranks = List(8, 12)
		val lambda = 0.1
		val numIters = List(10/*, 20*/)
		var bestModel: Option[MatrixFactorizationModel] = None
		var bestValidationRmse = Double.MaxValue
		var bestRank = 0
		var bestLambda = -1.0
		var bestNumIter = -1
		for(rank <- ranks; numIter <- numIters) {
			val model = ALS.train(training, rank, numIter, lambda)
			val validationsRmse = computeRmse(model, validation)
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
		
		val testRmse = computeRmse(bestModel.get, test)
		println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
				+ ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")
		
		sc.stop()
	}
	
	/** Compute RMSE (Root Mean Squared Error). */
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }
}

/*class MovieLen {
	def run():Double={
		Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
		Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)
		
		val master = "spark://slaver2:7077"
		val dataSrc= "hdfs://slaver2:9000/user/hadoop/MovieLens/ml-1m/"
		
		val conf = new SparkConf()
			.setMaster(master)
			.set("spark.executor.memory", "8G")
			.set("spark.driver.memory", "8G")
			.setAppName("MovieLensALS")
		val sc = new SparkContext(conf)
		
		val ratings = sc.textFile(dataSrc + "ratings.dat").map { line =>
			val fields = line.split("::")
			(fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
			}
		val movies  = sc.textFile(dataSrc + "movies.dat").map { line =>
			val fields = line.split("::")
			(fields(0).toInt, fields(1))
		}.collect.toMap
		
		println("current ok..")
		// TODO
		val numRatings = ratings.count()
		val numUser		 = ratings.map(_._2.user).distinct().count()
		val numMovies	 = ratings.map(_._2.product).distinct().count()
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
		
		val ranks = List(8, 12)
		val lambda = 0.1
		val numIters = List(10, 20)
		var bestModel: Option[MatrixFactorizationModel] = None
		var bestValidationRmse = Double.MaxValue
		var bestRank = 0
		var bestLambda = -1.0
		var bestNumIter = -1
		for(rank <- ranks; numIter <- numIters) {
			val model = ALS.train(training, rank, numIter, lambda)
			val validationsRmse = computeRmse(model, validation)
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
		
		val testRmse = computeRmse(bestModel.get, test)
		println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda
				+ ", and numIter = " + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")
		
		sc.stop()
		
		return testRmse
	}
	def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating]): Double = {

    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map{ x =>
      ((x.user, x.product), x.rating)
    }.join(data.map(x => ((x.user, x.product), x.rating))).values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).mean())
  }
}*/