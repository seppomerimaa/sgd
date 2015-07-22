package models

import org.apache.spark.mllib.linalg.Vectors
import scala.math.sqrt

/**
 * Fuck this Scala / MLLib / Breeze naming clusterfuck
 */
object VectorHelpers {
  implicit class VectorShadow(thisVector: org.apache.spark.mllib.linalg.Vector) {
    def dot(thatVector: org.apache.spark.mllib.linalg.Vector): Double = {
      (thisVector.toArray zip thatVector.toArray).map {
        case (v1: Double, v2: Double) => v1 * v2
      }.sum
    }

    def scaleBy(x: Double): org.apache.spark.mllib.linalg.Vector = {
      Vectors.dense(thisVector.toArray.map(v => v * x))
    }

    def plus(thatVector: org.apache.spark.mllib.linalg.Vector): org.apache.spark.mllib.linalg.Vector = {
      val raw = (thisVector.toArray zip thatVector.toArray).map {
        case (v1: Double, v2: Double) => v1 + v2
      }
      Vectors.dense(raw)
    }

    def minus(thatVector: org.apache.spark.mllib.linalg.Vector): org.apache.spark.mllib.linalg.Vector = {
      val raw = (thisVector.toArray zip thatVector.toArray).map {
        case (v1: Double, v2: Double) => v1 - v2
      }
      Vectors.dense(raw)
    }

    def magnitude: Double = {
      sqrt(thisVector.toArray.map(v => v * v).sum)
    }
  }
}
//
//object VectorHelpersTest extends App {
//  import models.VectorHelpers._
//
//  val v1 = Vectors.dense(1, 2, 3, 4)
//  val v2 = Vectors.dense(5, 6, 7, 8)
//
//  println(v1.dot(v2)) // 5 + 12 + 21 + 32 = 70
//  println(v1.scaleBy(2)) // [2, 4, 6, 8]
//  println(v1 plus v2) // [6, 8, 10, 12]
//  println(v2 minus v1) // [4, 4, 4, 4]
//  println(v1.magnitude) // sqrt(1 + 4 + 9 + 16) = sqrt(30) = 5.477225575051661
//}
