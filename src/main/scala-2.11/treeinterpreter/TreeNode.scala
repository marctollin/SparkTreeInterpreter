package org.apache.spark.ml.treeinterpreter

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tree.Node
import org.apache.spark.ml.tree.LeafNode
import org.apache.spark.ml.tree.InternalNode

object TreeNode {

  type NodeID = String
  type NodeMap = Map[NodeID, Double]

  trait SimplifiedNode {
    def NodeID: NodeID

    def predictLeaf(features: Vector)(implicit node2Treenode: Node => TreeNode): String

    def value: Double

    def feature: Option[Int]
  }

  abstract class TreeNode(node: Node) extends SimplifiedNode with Ordered[TreeNode] {
    override def toString(): String = Array(node.toString, value, feature).mkString("||", ",", "||")

    def compare(that: TreeNode) =  this.NodeID compare that.NodeID

    val NodeID: String = node.toString

    val feature: Option[Int] = node match {
      case node: InternalNode => Some(node.split.featureIndex)
      case _: LeafNode => None
    }

    def value: Double

    def predictLeaf(features: Vector)(implicit node2Treenode: Node => TreeNode): NodeID =
      node match {
        case _: LeafNode =>
          node.toString
        case node: InternalNode =>
          if (node.split.shouldGoLeft(features)) {
            node.leftChild.predictLeaf(features)
          } else {
            node.rightChild.predictLeaf(features)
          }
      }
  }

  case class ClassificationNode(node: Node) extends TreeNode(node: Node) {
    override def value: Double = node.impurityStats.prob(node.prediction) + .00001 // because algebird MapMonoid discards 0 values
  }

  case class RegressionNode(node: Node) extends TreeNode(node: Node) {
    override def value: Double = node.prediction
  }
}
