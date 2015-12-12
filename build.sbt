name := "SparkTreeInterpreter"

version := "1.0"

scalaVersion := "2.11.7"


libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.4.1",
  "org.apache.spark" %% "spark-mllib" % "1.4.1",
  "com.twitter" %% "algebird-core" % "0.9.0",
  "org.scalaz" %% "scalaz-core" % "7.1.0")
