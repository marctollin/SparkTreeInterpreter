import com.holdenkarau.spark.testing.LocalSparkContext
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Suite}
import org.apache.spark.ml.treeinterpreter.Interp
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{functions => F}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.RandomForestRegressionModel
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.classification.RandomForestClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.sql.{Dataset, Row}
import treeinterpreter.utils.IndicesToFeatures

class InterpTest extends FunSuite with SharedSparkSession {
  def prepareRegressorData(spark: SparkSession): (Dataset[Row], Dataset[Row]) = {
    import spark.implicits._    
    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/bostonData.data"
    val data = spark
      .read
      .option("header", true)
      .csv(replacementPath)
      .select(('A' to 'N').map {c => F.col(c.toString).cast("double")}:_*)

    val assembler = new VectorAssembler()
      .setInputCols(('A' until 'N').map{c => c.toString}.toArray)
      .setOutputCol("features")
    
    val featuredData = assembler
      .transform(data)
      .select($"features", $"N".as("label"))

    featuredData.printSchema()
    featuredData.show(5)

    val splits = featuredData.randomSplit(Array(0.2, 0.8), seed = 5)
    val (trainingData, testData) = (splits(0), splits(1))
    (trainingData, testData)
  }

  def prepareClassifierData(spark: SparkSession): (Dataset[Row], Dataset[Row]) = {
    val ss = spark
    import ss.implicits._
    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/bostonData.data"
    val data = spark
      .read
      .option("header", true)
      .csv(replacementPath)
      .select(('A' to 'N').map {c => F.col(c.toString).cast("double")}:_*)

    val assembler = new VectorAssembler()
      .setInputCols((Seq("A", "B", "C") ++ ('E' to 'N').map{c => c.toString}).toArray)
      .setOutputCol("features")
    
    val featuredData = assembler
      .transform(data)
      .select($"features", $"D".as("label"))

    featuredData.printSchema()
    featuredData.show(5)
  
    val splits = featuredData.randomSplit(Array(0.2, 0.8), seed = 5)
    val (trainingData, testData) = (splits(0), splits(1))
    (trainingData, testData)
  }

  test("Random Forest Regression Test") {
    val (trainingData, testData) = prepareRegressorData(spark)

    val numTrees = 10
    val featureSubsetStrategy = "auto"
    val maxDepth = 5
    val maxBins = 32
    val classimpurity = "variance"

    val rf = new RandomForestRegressor()
      .setSeed(1234)
      .setImpurity(classimpurity)
      .setMaxBins(maxBins)
      .setMaxDepth(maxDepth)
      .setFeatureSubsetStrategy(featureSubsetStrategy)
      .setNumTrees(numTrees)
    
    val model : RandomForestRegressionModel = rf.fit(trainingData)
    val predicted = model.transform(testData)
    val interpDataset = Interp.interpretModelTf(spark, model, testData)

    interpDataset.take(5).foreach(println)
    predicted.take(5).foreach(println)
    println(s"features importance: ${model.featureImportances}")
    println(s"indices to features ${IndicesToFeatures(trainingData)}")
  }

  test("Random Forest Classifier Test") {
    val (trainingData, testData) = prepareClassifierData(spark)

    val numTrees = 10
    val featureSubsetStrategy = "auto"
    val maxDepth = 5
    val maxBins = 32
    val classimpurity = "gini"

    val rf = new RandomForestClassifier()
      .setSeed(1234)
      .setImpurity(classimpurity)
      .setMaxBins(maxBins)
      .setMaxDepth(maxDepth)
      .setFeatureSubsetStrategy(featureSubsetStrategy)
      .setNumTrees(numTrees)
    
    val model : RandomForestClassificationModel = rf.fit(trainingData)
    model.transform(testData).show(5)
    val interpDataset = Interp.interpretModelTf(spark, model, testData)

    interpDataset.take(5).foreach(println)
    println(s"features importance: ${model.featureImportances}")
    println(s"indices to features ${trainingData.schema("features").metadata.getMetadata("ml_attr").getMetadata("attrs")}")
  }

  test("Decision Tree Classifier Test") {
    val (trainingData, testData) = prepareClassifierData(spark)

    val maxDepth = 5
    val maxBins = 32
    val classimpurity = "gini"

    val dt = new DecisionTreeClassifier()
      .setSeed(1234)
      .setImpurity(classimpurity)
      .setMaxBins(maxBins)
      .setMaxDepth(maxDepth)
    
    val model : DecisionTreeClassificationModel = dt.fit(trainingData)
    model.transform(testData).show(5)
    val interpDataset = Interp.interpretModelDt(spark, model, testData)

    interpDataset.take(10).foreach(println)
    println(s"features importance: ${model.featureImportances}")
    println(s"indices to features ${trainingData.schema("features").metadata.getMetadata("ml_attr").getMetadata("attrs")}")
  }

  test("Decision Tree Regression Test") {
    val (trainingData, testData) = prepareRegressorData(spark)

    val maxDepth = 5
    val maxBins = 32
    val classimpurity = "variance"

    val dt = new DecisionTreeRegressor()
      .setSeed(1234)
      .setImpurity(classimpurity)
      .setMaxBins(maxBins)
      .setMaxDepth(maxDepth)
    
    val model : DecisionTreeRegressionModel = dt.fit(trainingData)

    val interpDataset = Interp.interpretModelDt(spark, model, testData)

    println(s"features importance: ${model.featureImportances}")
    interpDataset.take(20).foreach(println)
  }
}

trait SharedSparkSession extends BeforeAndAfterAll {
  self: Suite =>

  @transient private var _ss: SparkSession = _

  implicit def spark: SparkSession = _ss

  val conf = new SparkConf().setMaster("local[*]")
    .setAppName("test")


  override def beforeAll() {
    _ss = SparkSession
          .builder
          .config(conf)
          .getOrCreate()
    super.beforeAll()
  }

  override def afterAll() {
    _ss.stop()
    _ss = null
    super.afterAll()
  }
}