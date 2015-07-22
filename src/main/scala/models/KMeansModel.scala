package models


import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.{sqrt, pow}

/**
 * A k-means cluster model, plus some utility functions.
 */
class KMeansModel(val centroids: List[LabeledPoint]) {
  def label(point: LabeledPoint): LabeledPoint = new LabeledPoint(KMeansModel.findNearestCentroid(centroids, point), point.features)
  def avgSquaredError(points: List[LabeledPoint]): Double = {
    val labelToCentroidMap: Map[Double, LabeledPoint] = centroids.map(c => (c.label, c)).toMap
    KMeansModel.avgSquaredError(labelToCentroidMap, points)
  }
}

object KMeansModel {
  def findNearestCentroid(centroids: List[LabeledPoint], point: LabeledPoint): Double = {
    val labelsAndDistances = centroids.map(c => (c.label, sqdist(c.features, point.features))).sortBy(p => p._2)
    labelsAndDistances.head._1
  }

  def euclideanDistance(v1: Vector[Double], v2: Vector[Double]): Double = {
    val total: Double = (v1 zip v2).map { case  (u1: Double, u2: Double) => pow(u1 + u2, 2) }.sum
    sqrt(total)
  }

  // We're only gonna be using this for small things, so the toArray bits are fine
  def sqdist(v1: org.apache.spark.mllib.linalg.Vector, v2: org.apache.spark.mllib.linalg.Vector): Double = {
    if (v1.size != v2.size) throw new IllegalArgumentException("Vector sizes don't match.")
    (v1.toArray zip v2.toArray).map { case (u1: Double, u2: Double) => pow(u1 - u2, 2)}.sum
  }

  def avg(vs: List[org.apache.spark.mllib.linalg.Vector]): org.apache.spark.mllib.linalg.Vector = {
    val a = new Array[Double](vs.head.size)
    val vsAsArrays: List[Array[Double]] = vs.map(v => v.toArray)
    a.indices.foreach { i =>
      a(i) = vsAsArrays.map(v => v(i)).sum / vs.size
    }
    Vectors.dense(a)
  }

  def weightedSubtract(v1: org.apache.spark.mllib.linalg.Vector, v2: org.apache.spark.mllib.linalg.Vector, weight: Double): org.apache.spark.mllib.linalg.Vector = {
    val rawPoints = (v1.toArray zip  v2.toArray).map {
      case (u1: Double, u2: Double) => (u1 - u2) * weight
    }
    Vectors.dense(rawPoints)
  }

  def add(v1: org.apache.spark.mllib.linalg.Vector, v2: org.apache.spark.mllib.linalg.Vector): org.apache.spark.mllib.linalg.Vector = {
    val rawPoints = (v1.toArray zip v2.toArray).map {
      case (u1: Double, u2: Double) => u1 + u2
    }
    Vectors.dense(rawPoints)
  }

  def avgSquaredError(labelToCentroidMap: Map[Double, LabeledPoint], points: List[LabeledPoint]): Double = {
    points.map(p => sqdist(p.features, labelToCentroidMap(p.label).features)).sum / points.size
  }
}

//object tests extends App {
//  override def main (args: Array[String]) {
//    val centroids = List(new LabeledPoint(1, Vectors.dense(2.0, 2.0)), new LabeledPoint(2, Vectors.dense(2, 7)), new LabeledPoint(3, Vectors.dense(7, 4)))
//    val v1 = new LabeledPoint(1, Vectors.dense(1, 1))
//    val v2 = new LabeledPoint(1, Vectors.dense(5, 2))
//    val v3 = new LabeledPoint(1, Vectors.dense(3, 6))
//
//    println(KMeansModel.sqdist(v1.features, v2.features)) // 17
//    println(KMeansModel.sqdist(v2.features, v3.features)) // 20
//    println(KMeansModel.sqdist(v1.features, v3.features)) // 29
//
//    println(KMeansModel.findNearestCentroid(centroids, v1)) // 1
//    println(KMeansModel.findNearestCentroid(centroids, v2)) // 3
//    println(KMeansModel.findNearestCentroid(centroids, v3)) // 2
//
//    val points = KMeansModeler.generateInitialCentroids(5, List((0.0, 10.0), (0.0, 10.0), (0.0, 10.0)))
//    points.map(println)
//
//    println(KMeansModel.avg(List(v1.features, v2.features, v3.features, Vectors.dense(10, -4)))) // 4.75, 1.25
//  }
//}
