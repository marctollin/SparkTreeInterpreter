name := "SparkTreeInterpreter"

version := "1.0.1"

scalaVersion := "2.10.5"

resolvers += Resolver.mavenLocal

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.6.0",
  "org.apache.spark" %% "spark-mllib" % "1.6.0",
  "com.twitter" %% "algebird-core" % "0.9.0",
  "org.scalaz" %% "scalaz-core" % "7.1.0",
  "org.scalatest" %% "scalatest"  % "2.2.4" % "test",
  "com.holdenkarau" %% "spark-testing-base" % "1.5.0_0.1.3",
  "org.json4s" %% "json4s-native" % "3.2.10",
  "org.json4s" %% "json4s-jackson" % "3.2.10",
  "org.json4s" % "json4s-ext_2.10" % "3.2.10"
)
