package graphs

import models.BatchPerceptronModelBuilder
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.jfree.chart.{ChartFrame, JFreeChart}
import org.jfree.chart.axis.{ValueAxis, NumberAxis}
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.renderer.xy.{XYItemRenderer, XYLineAndShapeRenderer}
import org.jfree.data.xy.DefaultXYDataset

/**
 * Created by smclaughlin on 7/21/15.
 */
class PerceptronGraph(points: List[LabeledPoint], w: Array[Double], xRange: (Double, Double), yRange: (Double, Double)) {
  val posPoints = points.filter(p => p.label > 0).map(p => p.features.toArray).toArray.transpose
  val negPoints = points.filter(p => p.label < 0).map(p => p.features.toArray).toArray.transpose
  val separationLine = Array(Array[Double](xRange._1, xRange._2), Array[Double](-1.0 * w(0) * xRange._1 / w(1), -1.0 * w(0) * xRange._2 / w(1)))

  def plot() = {
    val plot = new XYPlot()

    // Plot the points
    val dataset1 = new DefaultXYDataset
    dataset1.addSeries("+1", posPoints)
    dataset1.addSeries("-1", negPoints)
    val renderer1 = new XYLineAndShapeRenderer(false, true)   // Shapes only
    val domain1 = new NumberAxis("X")
    val range1 = new NumberAxis("Y")

    // Set the scatter data, renderer, and axis into plot
    plot.setDataset(1, dataset1)
    plot.setRenderer(1, renderer1)
    plot.setDomainAxis(1, domain1)
    plot.setRangeAxis(1, range1)

    // Map the scatter to the first Domain and first Range
    plot.mapDatasetToDomainAxis(1, 1)
    plot.mapDatasetToRangeAxis(1, 1)

    // Plot the line
    val dataset2 = new DefaultXYDataset
    dataset2.addSeries("Centroids", separationLine)
    val renderer2 = new XYLineAndShapeRenderer(true, false)   // Lines only

    // Set the line data, renderer, and axis into plot
    plot.setDataset(0, dataset2)
    plot.setRenderer(0, renderer2)
    plot.setDomainAxis(0, domain1)
    plot.setRangeAxis(0, range1)

    // Map the line to the second Domain and second Range
    plot.mapDatasetToDomainAxis(1, 1)
    plot.mapDatasetToRangeAxis(1, 1)

    // Create the chart with the plot and a legend
    val chart = new JFreeChart("Perceptron", JFreeChart.DEFAULT_TITLE_FONT, plot, true)
    val frame = new ChartFrame("Perceptron", chart)
    frame.pack()
    frame.setVisible(true)
  }
}

object PerceptronGraphApp extends App {
  override def main(args: Array[String]): Unit = {
    val data = List(
      new LabeledPoint(-1, Vectors.dense(2,1)),
      new LabeledPoint(-1, Vectors.dense(4,2)),
      new LabeledPoint(-1, Vectors.dense(3,1)),
      new LabeledPoint(-1, Vectors.dense(4,3)),
      new LabeledPoint(-1, Vectors.dense(5,3)),
      new LabeledPoint(-1, Vectors.dense(6,4)),
      new LabeledPoint(1, Vectors.dense(1,2)),
      new LabeledPoint(1, Vectors.dense(1,3)),
      new LabeledPoint(1, Vectors.dense(2,4)),
      new LabeledPoint(1, Vectors.dense(2,5)),
      new LabeledPoint(1, Vectors.dense(3,5)),
      new LabeledPoint(1, Vectors.dense(4,6))
    )

    val onlinePerceptron = BatchPerceptronModelBuilder.build(1, 1000, data)

    //val w = Array(-1.0, 1.0)
    val xRange = (0.0, 10.0)
    val yRange = (0.0, 10.0)
    val graph = new PerceptronGraph(data, onlinePerceptron.w.toArray, xRange, yRange)
    graph.plot()
  }
}