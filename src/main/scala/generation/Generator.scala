package generation

import org.apache.commons.math3.distribution.NormalDistribution
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.util.Random
import scala.math

/**
 * Generates various sets of sample data.
 */
object Generator {
  def clusters(
                numClusters: Int,
                numPointsPerCluster: Int,
                mean: Double,
                variance: Double,
                xRange: (Double, Double),
                yRange: (Double, Double)
                ): List[LabeledPoint] = {
    val distr = new NormalDistribution(mean, variance)
    val centroids: Seq[(Double, Double)] = for (i <- 1 to numClusters) yield { (scale(Random.nextDouble(), xRange), scale(Random.nextDouble(), yRange) ) }
    val data = for (
      centroid <- centroids;
      i <- 1 to numPointsPerCluster
    ) yield {
        val xCoord = math.min(xRange._2, math.max(xRange._1, distr.sample() + centroid._1))
        val yCoord = math.min(yRange._2, math.max(yRange._1, distr.sample() + centroid._2))
      new LabeledPoint(0, Vectors.dense(xCoord, yCoord))
    }

    data.toList
  }

  def points(numPoints: Int, ranges: List[(Double, Double)]): List[List[Double]] = {
    (0 until numPoints).map(i => ranges.map(range => scale(Random.nextDouble(), range))).toList
  }

  private def scale(v: Double, range: (Double, Double)) = v * (range._2 - range._1) + range._1
}
