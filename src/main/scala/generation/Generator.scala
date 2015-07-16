package generation

import org.apache.commons.math3.distribution.NormalDistribution

import scala.util.Random

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
                ): List[(Double, Double)] = {
    val distr = new NormalDistribution(mean, variance)
    val centroids: Seq[(Double, Double)] = for (i <- 1 to numClusters) yield { (scale(Random.nextDouble(), xRange), scale(Random.nextDouble(), yRange) ) }
    val data: Seq[(Double, Double)] = for (
      centroid <- centroids;
      i <- 1 to numPointsPerCluster
    ) yield {
        (distr.sample() + centroid._1, distr.sample() + centroid._2)
    }

    data.toList
  }

  private def scale(v: Double, range: (Double, Double)) = v * (range._2 - range._1) + range._1
}
