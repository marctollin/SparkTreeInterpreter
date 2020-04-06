package org.apache.spark.ml.treeinterpreter

import com.twitter.algebird
import com.twitter.algebird.Monoid
import com.twitter.algebird.Operators._
import org.apache.spark.ml.tree.{DecisionTreeModel, TreeEnsembleModel}
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.treeinterpreter.DressedTree._
import org.apache.spark.sql.{functions => F}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Row
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.linalg.DenseVector


case class Interp(bias: Double,
                  prediction: Double,
                  contributions: Map[Feature, Double],
                  treeCount: Double = 1.0,
                  checksum: Double = 0.0) {
  override def toString(): String = {
    s"""
   | bias: $bias
   | prediction: $prediction
   | contributionMap $contributions
   | sumOfTerms: $checksum
   | tree count: $treeCount""".stripMargin
  }
}

object Interp {

  class InterpMonoid extends Monoid[Interp] {
    def plus(l: Interp, r: Interp) = new Interp(
      l.bias + r.bias,
      l.prediction + r.prediction,
      l.contributions + r.contributions,
      l.treeCount + r.treeCount)

    def zero = Interp(0.0, 0.0, Map())
  }

  implicit def InterpMonoidImpl: algebird.Monoid[Interp] = new InterpMonoid

  def interpretModelTf[M <: TreeEnsembleModel[_ <: DecisionTreeModel with PredictionModel[Vector, _]] with PredictorParams](
                     spark: SparkSession,
                     rf: M,
                     testSet: Dataset[Row]): Dataset[Interp] = {
    import spark.implicits._
    val trees = rf.trees.map(tree => DressedTree.trainInterpreter(tree))

    val result = testSet
      .select(F.col(rf.getFeaturesCol).as("features"))
      .map { row =>
        val vec = row.getAs[DenseVector]("features")
        trees.map {
          case dressedTree =>
          dressedTree.interpret(vec)
        }
      }

    val aggResult: Dataset[Interp] = result
      .map(_.reduce(_ + _))
      .map { case interp => {
        import interp._
        val (avgBias, avgPrediction) = (bias / treeCount, prediction / treeCount)

        val avgContributions = contributions.mapValues(_/treeCount).map(identity)

        val checkSum = avgBias + avgContributions.values.sum

        Interp(avgBias, avgPrediction, avgContributions, treeCount, checkSum)
      }
    }
    aggResult
  }

  def interpretModelDt[M <: DecisionTreeModel with PredictionModel[Vector, _] with PredictorParams](spark: SparkSession,
                     model: M,
                     testSet: Dataset[Row]): Dataset[Interp] = {
    import spark.implicits._
    val dressedTree = DressedTree.trainInterpreter(model)
    testSet
    .select(F.col(model.getFeaturesCol).as("features"))
    .map { row =>
      val vec = Vectors.dense(row.getAs[Array[Double]]("features"))
      dressedTree.interpret(vec)
    }
  }
}
