name := "SparkTreeInterpreter"

version := "1.2.1"

scalaVersion := "2.11.12"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.3.0",
  "org.apache.spark" %% "spark-mllib" % "2.3.0",
  "com.twitter" %% "algebird-core" % "0.13.0",
  "org.scalaz" %% "scalaz-core" % "7.2.30",
  "org.scalatest" %% "scalatest"  % "3.0.0" % "test",
  "com.holdenkarau" %% "spark-testing-base" % "2.4.3_0.14.0",
  "org.json4s" %% "json4s-native" % "3.4.2",
  "org.json4s" %% "json4s-jackson" % "3.4.2",
  "org.json4s" % "json4s-ext_2.11" % "3.4.2"
)
