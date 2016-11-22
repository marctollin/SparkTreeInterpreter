package treeinterpreter

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.tree.model.Node

object TreeNode {

  type NodeID = Int
  type NodeMap = Map[NodeID, Double]

  trait SimplifiedNode {
    def NodeID: NodeID

    def predictLeaf(features: Vector)(implicit node2Treenode: Node => TreeNode): Int

    def value: Double

    def feature: Option[Int]
  }

  abstract class TreeNode(node: Node) extends SimplifiedNode with Ordered[TreeNode] {
    import node._

    override def toString(): String = Array(id, value, feature).mkString("||", ",", "||")

    def compare(that: TreeNode) =  this.NodeID compare that.NodeID

    val NodeID: Int = id

    val feature: Option[Int] = split.map(_.feature)

    def value: Double

    def predictLeaf(features: Vector)(implicit node2Treenode: Node => TreeNode): NodeID = {
      if (isLeaf) {
        id
      } else {
        if (split.get.featureType == Continuous) {
          if (features(split.get.feature) <= split.get.threshold) {
            leftNode.get.predictLeaf(features)
          } else {
            rightNode.get.predictLeaf(features)
          }
        } else {
          if (split.get.categories.contains(features(split.get.feature))) {
            leftNode.get.predictLeaf(features)
          } else {
            rightNode.get.predictLeaf(features)
          }
        }
      }
    }
  }

  case class ClassificationNode(node: Node) extends TreeNode(node: Node) {
    override def value: Double = node.predict.prob + .00001 // because algebird MapMonoid discards 0 values
  }

  case class RegressionNode(node: Node) extends TreeNode(node: Node) {
    override def value: Double = node.predict.predict
  }
}
