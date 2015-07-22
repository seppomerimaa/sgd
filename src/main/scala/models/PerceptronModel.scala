package models

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import VectorHelpers._
import scala.util.Random


/**
 * Created by smclaughlin on 7/21/15.
 */
class PerceptronModel(val w: org.apache.spark.mllib.linalg.Vector) {
  def classify(point: org.apache.spark.mllib.linalg.Vector): LabeledPoint = {
    val rawLabel = point dot  w
    val label = if (rawLabel < 0) -1 else 1
    new LabeledPoint(label, point)
  }
}

/**
 * batchSize = 1 and numIter = 1 gets you an online / SGD perceptron
 * batchSize = num samples and numIter = w/e gets you a classic gradient descent perceptron
 */
object BatchPerceptronModelBuilder {
  private val tolerance = 0.0001
  private val trainingRate = 0.001
  def build(batchSize: Int, numIter: Int, points: List[LabeledPoint]): PerceptronModel = {
    var w = Vectors.dense(Array.fill[Double](points(0).productArity)(1))
    var accumulatedGradient = Vectors.dense(Array.fill[Double](points.head.productArity)(0))
    var counter = 0
    (1 to numIter).foreach { i =>
      val batch = Random.shuffle(points).take(batchSize)
      batch.foreach {point =>
        val u = point.features dot w
        //println(s"# p: ${point.features} w: $w label: ${point.label} u: $u")
        if (u * point.label < 0) {
          accumulatedGradient = accumulatedGradient plus point.features.scaleBy(point.label)
          //println("accumulated gradient: " + accumulatedGradient)
        }
      }
      w = w.plus(accumulatedGradient.scaleBy(trainingRate))
      counter += 1
      //println(s"magnitude: ${accumulatedGradient.magnitude} error: ${accumulatedGradient.magnitude / batchSize}")
    } //while(accumulatedGradient.magnitude / batchSize > tolerance && counter < numIter)
    println("w: " + w)
    new PerceptronModel(w)
  }
}
