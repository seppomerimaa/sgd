package graphs

import org.apache.spark.mllib.regression.LabeledPoint
import org.jfree.chart.{ChartFactory, ChartFrame}
import org.jfree.data.xy.DefaultXYDataset

/**
 * Created by smclaughlin on 7/21/15.
 */
class ClusterGraph(centroidLPs: List[LabeledPoint], points: List[List[LabeledPoint]]) {
  val centroids: Array[Array[Double]] = centroidLPs.map { lp =>
    require(lp.productArity == 2, "Got a labeled point whose arity wasn't 2...")
    lp.features.toArray
    //Array(lp.features))
  }.toArray.transpose
  val series = points.map { clusterPoints =>
    clusterPoints.map { point =>
      require(point.productArity == 2, "Got a labeled point whose arity wasn't 2...")
      point.features.toArray
    }.toArray.transpose
  }.toArray

  def plot() = {
    val dataset = new DefaultXYDataset
    dataset.addSeries("Centroids", centroids)
    series.indices.foreach(i => dataset.addSeries(s"Cluster $i", series(i)))
    val frame = new ChartFrame(
      "Title",
      ChartFactory.createScatterPlot(
        "Clusters",
        "X",
        "Y",
        dataset,
        org.jfree.chart.plot.PlotOrientation.VERTICAL,
        true,false,false
      )
    )
    frame.pack()
    frame.setVisible(true)
  }
}
