import com.holdenkarau.spark.testing.LocalSparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite, Suite}
import treeinterpreter.Interp

class InterpTest extends FunSuite with SharedSparkContext {
  test("Random Forest Regression Test") {

    implicit val _sc = sc

    val currentDir = System.getProperty("user.dir")
    val replacementPath = s"$currentDir/src/test/resources/bostonData.data"
    val data = _sc.textFile(replacementPath)

    val parsedData = data.map(line => {
      val parsedLine = line.split(',')
      LabeledPoint(parsedLine.last.toDouble, Vectors.dense(parsedLine.dropRight(1).map(_.toDouble)))
    })

    val splits = parsedData.randomSplit(Array(0.2, 0.8), seed = 5)
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 10
    val featureSubsetStrategy = "auto"
    val impurity = "variance"
    val maxDepth = 2
    val maxBins = 32

    val classimpurity = "gini"

    val rf = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed = 21)

        val labelsAndPredictions = trainingData.map { point =>
          val prediction = rf.predict(point.features)
          (point.label, prediction)
        }

    val testMSE = math.sqrt(labelsAndPredictions.map { case (v, p) => math.pow(v - p, 2) }.mean())

    println("Test Mean Squared Error = " + testMSE)

    val interpRDD = Interp.interpretModel(rf, trainingData)

    interpRDD.collect().foreach(println)

    interpRDD.collect().foreach(item=> assert(scala.math.abs(item.checksum/item.prediction-1)<.2))
  }
}

trait SharedSparkContext extends BeforeAndAfterAll {
  self: Suite =>

  @transient private var _sc: SparkContext = _

  implicit def sc: SparkContext = _sc

  val conf = new SparkConf().setMaster("local[*]")
    .setAppName("test")


  override def beforeAll() {
    _sc = new SparkContext(conf)
    super.beforeAll()
  }

  override def afterAll() {
    LocalSparkContext.stop(_sc)
    _sc = null
    super.afterAll()
  }
}