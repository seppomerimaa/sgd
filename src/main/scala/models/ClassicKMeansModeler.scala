package models

import generation.Generator
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.abs
import scala.util.control.Breaks._
/**
 * Created by McFly on 7/20/15.
 */
object ClassicKMeansModeler {
  val numIter = 1000
  val tol = 0.001

  def build(k: Int, data: List[LabeledPoint], ranges: List[(Double, Double)]): KMeansModel = {
    var centroids = generateInitialCentroids(k, ranges)
    var labelToCentroidMap: Map[Double, LabeledPoint] = centroids.map(c => (c.label, c)).toMap
    var prevError = Double.MaxValue

    breakable {
      for (i <- 0 until numIter) {
        // assign points to centroids
        val data2 = data.map(p => new LabeledPoint(KMeansModel.findNearestCentroid(centroids, p), p.features))

        // re-calculate centroids
        centroids = data2.groupBy(d => d.label).map { case (label: Double, points: List[LabeledPoint]) =>
          val newCentroid = KMeansModel.avg(points.map(p => p.features))
          new LabeledPoint(label, newCentroid)
        }.toList
        labelToCentroidMap = centroids.map(c => (c.label, c)).toMap

        // calculate error
        val error = KMeansModel.error(labelToCentroidMap, data2)
        if (abs(prevError - error) < tol) break()
      }
    }
    new KMeansModel(centroids)
  }

  def generateInitialCentroids(k: Int, ranges: List[(Double, Double)]): List[LabeledPoint] = {
    val points = Generator.points(k, ranges)
    points.zipWithIndex.map(pair => new LabeledPoint(pair._2, Vectors.dense(pair._1.toArray)))
  }
}
