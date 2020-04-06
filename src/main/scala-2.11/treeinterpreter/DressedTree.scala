package org.apache.spark.ml.treeinterpreter

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.{DecisionTreeModel, Node}
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.treeinterpreter.TreeNode._
import org.apache.spark.ml.treeinterpreter.DressedTree._
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.tree.LeafNode
import org.apache.spark.ml.tree.InternalNode
import org.apache.spark.ml.PredictorParams

case class DressedTree[M <: DecisionTreeModel with PredictionModel[Vector, _] with PredictorParams](model: M,
                       bias: Double,
                       contributionMap: NodeContributions) {

  implicit def nodeType(node: Node): TreeNode = model match {
    case _: DecisionTreeClassificationModel => ClassificationNode(node)
    case _: DecisionTreeRegressionModel => RegressionNode(node)
  }

  private def predictLeafID(features: Vector): NodeID = model.rootNode.predictLeaf(features)

  def predictLeaf(point: Vector): Interp = {
    val leaf = predictLeafID(point)
    val prediction = model.predict(point)
    val contribution = contributionMap(leaf)
    Interp(bias, prediction, contribution)
  }

  def interpret(point: Vector) = predictLeaf(point)
}


object DressedTree {

  type Feature = Option[Int]

  type NodeContributions = Map[NodeID, Map[Feature, Double]]

  def arrayprint[A](x: Array[A]): Unit = println(x.deep.mkString("\n"))

  def trainInterpreter[M <: DecisionTreeModel with PredictionModel[Vector, _] with PredictorParams](model: M): DressedTree[M] = {
    val topNode = model.rootNode

    implicit def nodeType(node: Node): TreeNode = model match {
      case _: DecisionTreeClassificationModel => ClassificationNode(node)
      case _: DecisionTreeRegressionModel => RegressionNode(node)
    }
    type Path = Array[TreeNode]

    type PathBundle = Array[Path]

    val bias = topNode.value

    var Paths: PathBundle = Array()

    def buildPath(paths: Path, node: Node): PathBundle = {
      val treeNode = nodeType(node)
      node match {
        case node: LeafNode =>
          Array(paths :+ treeNode)
        case node: InternalNode =>
          val buildRight = buildPath(paths :+ treeNode, node.rightChild)
          val buildLeft = buildPath(paths :+ treeNode, node.leftChild)
          buildRight ++ buildLeft
      }
    }

    val paths = buildPath(Array(), topNode).map(_.reverse)
    DressedTree.arrayprint(paths)

    val contributions: NodeContributions = paths.flatMap(path => {
      val contribMap = path
        .zip(path.tail)
        .flatMap {
          case (currentNode, prevNode) =>
            Map(prevNode.feature -> {
              currentNode.value - prevNode.value
            })
        }
        .foldLeft(Map.empty[Feature, Double])(_ + _)

      val leafID = path.head.nodeID
      Map(leafID -> contribMap)
    }).toMap

    DressedTree(model, bias, contributions)
  }
}
