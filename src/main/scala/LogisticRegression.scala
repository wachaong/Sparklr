import java.io.PrintWriter
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{norm => brzNorm}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS}
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._

import org.apache.log4j._

import scala.collection.immutable.SortedMap

case class ClkImpInstance(var click: Int, var impression: Int, var features: BSV[Double])

class LRWithLBFGS(val l2RegParam: Double,
				  val memParam: Int,
				  val maxNumIterations: Int,
				  val tolerance: Double) extends Serializable {

  def train(trainSet: RDD[ClkImpInstance], initialWeights: BDV[Double]) = {

	val costFun = new CostFun(trainSet, l2RegParam)

	val lbfgs = new LBFGS[BDV[Double]](maxNumIterations, memParam, tolerance)

	val states = lbfgs
	  .iterations(new CachedDiffFunction(costFun), initialWeights)

	var state = states.next()
	while (states.hasNext) {
	  state = states.next()
	  println("iter:"+state.iter)
	}
	val weights = state.x
	
	weights
  }

  private class CostFun(data: RDD[ClkImpInstance],
						l2RegParam: Double) extends DiffFunction[BDV[Double]] with Serializable{

	def calGradientLossInstance(labeledInstance: ClkImpInstance,
								weightsBC: Broadcast[BDV[Double]]): (BSV[Double], Double) = {
	

	  def score2LossProb(score: Double) = {
		if (score < -30) {
		  (-score, 0.0)
		} else if (score > 30) {
		  (0.0, 1.0)
		} else {
		  val tmp = 1 + math.exp(-score)
		  (math.log(tmp), 1.0 / tmp)
		}
	  }

          val logger = Logger.getRootLogger()
	  val weights = weightsBC.value
	  val (clicks, imps) = (labeledInstance.click, labeledInstance.impression)
	  val nonclicks = imps - clicks

	  val x = labeledInstance.features

	  val score = x dot weights

	  var totalMult = 0.0
	  var totalLoss = 0.0

	  if (clicks > 0) {
		val (loss, prob) = score2LossProb(score)
		val mult = (prob - 1.0) * clicks
		totalMult = mult
		totalLoss = loss * clicks
	  }

	  if (nonclicks > 0) {
		val (loss, prob) = score2LossProb(-1*score)
		val mult = (1.0 - prob) * nonclicks
		totalMult += mult
		totalLoss += loss * nonclicks
	  }

	  val gradient = x * totalMult
	  (gradient, totalLoss)
	}

        override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
          val wb = data.sparkContext.broadcast(x)

          val kvs = data.flatMap(inst => {
                val (grad, loss) = calGradientLossInstance(inst, wb)
                grad.activeIterator.toSeq :+(-1, loss)
          }).reduceByKey(_ + _,100)

          val grad = kvs.filter(_._1>=0).map(inst=>{
                (inst._1,inst._2 + l2RegParam * x(inst._1))
          }).collect

	  val gradient = new Array[Double](x.length)
	  grad.map(inst =>{
		gradient(inst._1) = inst._2
	  })
	  //gradient.foreach(println)

          val norm = brzNorm(x,2)
	  val tmp = kvs.filter(inst => inst._1 == -1).first._2.toDouble
          val loss = tmp + 0.5 * l2RegParam * norm * norm

	  //println("######################" + tmp,loss,norm)
          (loss, new BDV(gradient))
        }

	/*override def calculate(x: BDV[Double]): (Double, BDV[Double]) = {
	  val wb = data.sparkContext.broadcast(x)

	  val kvs = data.flatMap(inst => {
		val (grad, loss) = calGradientLossInstance(inst, wb)
		grad.activeIterator.toSeq :+(-1, loss)
	  }).reduceByKeyLocally(_ + _)

	  val gradient = new Array[Double](x.length)
	  for (index <- kvs.filterKeys(_ >= 0).keys)
		gradient(index) = kvs.getOrElse(index, 0.0) + l2RegParam * x(index)
	  val norm = brzNorm(x,2)
	  val loss = kvs(-1) + 0.5 * l2RegParam * norm * norm
	  (loss, new BDV(gradient))
	}*/

  }

}

object LogisticRegression {

  def line2Vector(content:String, length: Int) = {
	val indices = new ArrayBuffer[Int]()
	val values = new ArrayBuffer[Double]()

	for(field <- content.split("\\p{Blank}+")) {
	  val kv = field.split(":")
	  if(kv.length == 2) {
		indices += kv(0).toInt
		values += kv(1).toDouble
	  }
	}

	val (sortedIndices,sortedValues) = (indices zip values).sortBy(_._1).unzip
	val bsv = new BSV[Double](sortedIndices.toArray,sortedValues.toArray,length)

	val addIntercept = BSV.vertcat(bsv,new BSV[Double](Array(0), Array(1.0), 1))

	addIntercept
  }

  def line2ClkImpInstance(content:String,length:Int): ClkImpInstance = {

	val items = content.split("\\p{Blank}+",3)
	val clk = items(0).toInt
	val imp = items(1).toInt
	val features = items(2)

	ClkImpInstance(clk, imp, line2Vector(features,length))
  }

  def main(args: Array[String]): Unit = {
	val logger = Logger.getRootLogger()

        val input = args(0)
        val length = args(1).toInt
        val iterNum = args(2).toInt

	val conf = new SparkConf().setAppName("LRTrain")
	val sc = new SparkContext(conf)

	val initWeights = BDV.zeros[Double](length + 1)

	val rawText = sc.textFile(input)
	val trainset = rawText.map(x=> line2ClkImpInstance(x,length)).cache()

	val model = new LRWithLBFGS(1,10,iterNum,1E-9).train(trainset,initWeights)
	//val model = new LRWithLBFGS(100,10,iterNum,1E-9).train(trainset,initWeights)
	
        var map = SortedMap[Int,String]()
        for (index <- model.keysIterator) {
                map += index -> (index+"\t"+model(index).toString)
        }
        val arr = map.values.toArray
        val t = sc.parallelize(arr)
        t.saveAsTextFile(args(3))

	sc.stop()
  }
}

