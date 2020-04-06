package treeinterpreter.utils


import org.json4s.jackson.Serialization._

trait toJsonString {
  override def toString = {
    implicit val formats = org.json4s.DefaultFormats
    write(this)
  }
}