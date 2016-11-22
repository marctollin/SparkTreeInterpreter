package treeinterpreter

import com.twitter.algebird
import com.twitter.algebird.Monoid
import com.twitter.algebird.Operators._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.model.{DecisionTreeModel, RandomForestModel}
import org.apache.spark.rdd.RDD
import treeinterpreter.DressedTree.Feature

case class Interp(bias: Double, prediction: Double, contributions: Map[Feature, Double], treeCount: Double = 1.0, checksum: Double = 0.0) {
  override def toString(): String = {
    s"""
   | bias: $bias
   | prediction: $prediction
   | contributionMap $contributions
   | sumOfTerms: $checksum""".stripMargin
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

  def interpretModel(rf: RandomForestModel, testSet: RDD[LabeledPoint]): RDD[Interp] = {
    val trees = rf.trees.map(tree => DressedTree.trainInterpreter(tree))

    val result = testSet.map(lp => {
      trees.map { case dressedTree =>
        dressedTree.interpret(lp.features)
      }
    })

    val aggResult: RDD[Interp] = result
      .map(_.reduce(_ + _))
      .map { case interp => {
        import interp._
        val (avgBias, avgPrediction) = (bias / treeCount, prediction / treeCount)

        val avgContributions = contributions.mapValues(_/treeCount).map(identity)

        val checkSum = avgBias + avgContributions.values.sum

        Interp(avgBias, avgPrediction, avgContributions, treeCount,checkSum)
      }
    }
    aggResult
  }

  def interpretModel(model: DecisionTreeModel, testSet: RDD[LabeledPoint]): RDD[Interp] = {
    val dressedTree = DressedTree.trainInterpreter(model)
    testSet.map(lp => dressedTree.interpret(lp.features))
  }
}