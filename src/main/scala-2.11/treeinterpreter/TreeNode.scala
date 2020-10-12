package org.apache.spark.ml.treeinterpreter

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.Node
import org.apache.spark.ml.tree.LeafNode
import org.apache.spark.ml.tree.InternalNode

object TreeNode {

  type NodeID = (Int, Double)

  type Feature = Option[(Int, Double)]

  trait SimplifiedNode {
    def nodeID: NodeID

    def predictLeaf(features: Vector)(implicit node2Treenode: Node => TreeNode): NodeID

    def value: Double

    def feature: Feature
  }

  abstract class TreeNode(node: Node) extends SimplifiedNode {
    override def toString(): String = Array(nodeID, value).mkString("||", ",", "||")

    val feature: Feature = node match {
      case node: InternalNode => Some(node.split.featureIndex, node.impurity)
      case _: LeafNode => Some(-1, node.impurity)
    }

    val nodeID: NodeID = feature.get

    def value: Double

    def predictLeaf(features: Vector)(implicit node2Treenode: Node => TreeNode): NodeID =
      node match {
        case node: LeafNode =>
          node.nodeID
        case node: InternalNode =>
          if (node.split.shouldGoLeft(features)) {
            node.leftChild.predictLeaf(features)
          } else {
            node.rightChild.predictLeaf(features)
          }
      }
  }

  case class ClassificationNode(node: Node) extends TreeNode(node) {
    override def value: Double = node.impurityStats.prob(node.prediction) + .00001 // because algebird MapMonoid discards 0 values
  }

  case class RegressionNode(node: Node) extends TreeNode(node) {
    override def value = node.prediction
  }
}
