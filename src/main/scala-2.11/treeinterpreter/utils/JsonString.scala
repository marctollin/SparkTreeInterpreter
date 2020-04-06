package treeinterpreter.utils


import org.json4s.jackson.Serialization._
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.attribute.AttributeKeys
import org.apache.spark.ml.attribute.AttributeType

trait toJsonString {
  override def toString = {
    implicit val formats = org.json4s.DefaultFormats
    write(this)
  }
}

object IndicesToFeatures {
  def apply(dataset: Dataset[_]): Map[Long, String] = {
    def toMap(attrType: AttributeType) = {
      val metaML = dataset
      .schema("features")
      .metadata
      .getMetadata("ml_attr")
      .getMetadata("attrs")
      if (!metaML.contains(attrType.name))
        Map.empty
      else {
        metaML
        .getMetadataArray(attrType.name)
        .map { e =>
          val idx = e.getLong("idx")
          val name = e.getString("name")
          (idx, name)
        }.toMap
      }
    }
    toMap(AttributeType.Numeric) ++ toMap(AttributeType.Binary) ++ toMap(AttributeType.Nominal)
  }
}