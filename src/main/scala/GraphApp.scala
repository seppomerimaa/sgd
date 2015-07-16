import generation.Generator
import org.jfree.chart.{ChartFrame, ChartFactory}
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.DefaultXYDataset

import scalax.chart.XYChart
import scalax.chart.module.Charting
import scalax.chart.api._

/**
 * Created by McFly on 7/15/15.
 */
object GraphApp extends App with Charting {
  val data = Generator.clusters(4, 10, 0, .2, (0, 10), (0, 10))
  //val chart = XYLineChart(data)
  //val chart2 = XYChart(data)
  val dataset = new DefaultXYDataset
  dataset.addSeries("foo", convertToGraphableData(data))
  val frame = new ChartFrame(
    "Title",
    ChartFactory.createScatterPlot(
      "Plot",
      "X Label",
      "Y Label",
      dataset,
      org.jfree.chart.plot.PlotOrientation.HORIZONTAL,
      false,false,false
    )
  )
  frame.pack()
  frame.setVisible(true)


  //val chart2 = ChartFactory.createScatterPlot("foo", "X", "Y", ToXYDataset.FromTuple2s(data))
  //ToXYDataset.FromTuple2s(data)
  //chart2.show()
  def convertToGraphableData(data: Seq[(Double, Double)]) = {
    data.toArray.map(d => Array(d._1, d._2)).transpose
  }
}
