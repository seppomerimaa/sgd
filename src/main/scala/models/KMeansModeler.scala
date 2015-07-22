package models

import generation.Generator
import graphs.ClusterGraph
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.collection.mutable
/**
 * Created by McFly on 7/20/15.
 */
object KMeansModeler {
  val numIter = 1000
  val tol = 0.001

  def buildClassic(k: Int, data: List[LabeledPoint], ranges: List[(Double, Double)]): KMeansModel = {
    var centroids = generateInitialCentroids(k, ranges)
    var labelToCentroidMap: Map[Double, LabeledPoint] = centroids.map(c => (c.label, c)).toMap

      for (i <- 0 until numIter) {
        // assign points to centroids
        val data2 = data.map(p => new LabeledPoint(KMeansModel.findNearestCentroid(centroids, p), p.features))

        // re-calculate centroids
        val stuff = data2.groupBy(d => d.label)
        centroids = stuff.map { case (label: Double, points: List[LabeledPoint]) =>
          //println(i + " " + label)
          val newCentroid = KMeansModel.avg(points.map(p => p.features))
          new LabeledPoint(label, newCentroid)
        }.toList
        labelToCentroidMap = centroids.map(c => (c.label, c)).toMap
      }
    new KMeansModel(centroids)
  }

  def buildOnline(k: Int, data: List[LabeledPoint], ranges: List[(Double, Double)]): KMeansModel = {
    val centroids: Array[LabeledPoint] = generateInitialCentroids(k, ranges).toArray // indices *should* match labels...
    // keep track of centroids & their counts
    val labelsToCounts = mutable.Map[Double, Int]()
    val labelsToCentroids = mutable.Map[Double, LabeledPoint]()
    centroids.foreach { c =>
      labelsToCounts(c.label) = 1
      labelsToCentroids(c.label) = c
    }
    // for each point, find nearest centroid, update centroid, and update centroid count
    data.map {point =>
      val closestLabel = KMeansModel.findNearestCentroid(labelsToCentroids.values.toList, point)
      labelsToCounts(closestLabel.toInt) += 1
      val originalCentroid = centroids(closestLabel.toInt)
      centroids(closestLabel.toInt) = updateCentroid(originalCentroid, point, 1.0 / labelsToCounts(closestLabel))
    }
    new KMeansModel(centroids.toList)
  }

  def updateCentroid(centroid: LabeledPoint, point: LabeledPoint, eps: Double): LabeledPoint = {
    val p = point.features
    val c = centroid.features
    val update = KMeansModel.weightedSubtract(p, c, eps)
    val updatedCentroid = KMeansModel.add(c, update)
    new LabeledPoint(centroid.label, updatedCentroid)
  }

  def generateInitialCentroids(k: Int, ranges: List[(Double, Double)]): List[LabeledPoint] = {
    val points = Generator.points(k, ranges)
    points.zipWithIndex.map(pair => new LabeledPoint(pair._2, Vectors.dense(pair._1.toArray)))
  }
}

/**
 * k = 10 & points / cluster = 500 works in < 30 seconds or so...
 */
object ClassicKMeansApp extends App {
  override def main(args: Array[String]): Unit = {
    val k = 5
    val numPointsPerCluster = 2000
    val xRange = (0.0, 15000.0)
    val yRange = (0.0, 15000.0)
    val points = Generator.clusters(k, numPointsPerCluster, 0, 1000.0, xRange, yRange)
    println(s"k: $k points per cluster: $numPointsPerCluster")
    val start = System.currentTimeMillis()
    val model = KMeansModeler.buildClassic(k, points, List(xRange, yRange))
    println(s"time to model: ${(System.currentTimeMillis() - start).toDouble / 1000}s")
    println(s"avg. squared error: ${model.avgSquaredError(points)}")
    val labeledPoints = points.map(p => model.label(p))
    val stuff = labeledPoints.groupBy(p => p.label).map(pair => pair._2).toList
    val graph = new ClusterGraph(model.centroids, stuff)
    graph.plot()
  }
}

object OnlineKmeansApp extends App {
  override def main(args: Array[String]): Unit = {
    val k = 5
    val numPointsPerCluster = 2000
    val xRange = (0.0, 15000.0)
    val yRange = (0.0, 15000.0)
    val points = Generator.clusters(k, numPointsPerCluster, 0, 1000.0, xRange, yRange)
    println(s"k: $k points per cluster: $numPointsPerCluster")
    val start = System.currentTimeMillis()
    val model = KMeansModeler.buildOnline(k, points, List(xRange, yRange))
    println(s"time to model: ${(System.currentTimeMillis() - start).toDouble / 1000}s")
    println(s"avg. squared error: ${model.avgSquaredError(points)}")
    val labeledPoints = points.map(p => model.label(p))
    val stuff = labeledPoints.groupBy(p => p.label).map(pair => pair._2).toList
    val graph = new ClusterGraph(model.centroids, stuff)
    graph.plot()
  }
}
