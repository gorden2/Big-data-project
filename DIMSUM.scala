import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}
import org.apache.spark.{SparkConf, SparkContext}

/**
Sample input data: userid,businessid,star
Output: userid1,userid2,similarity(cosine)
*/
val data = sc.textFile("review5w")
val users = data.map(l => (l.split(",")(0),l.split(",")(1),l.split(",")(2)))
val userset = data.map(_.split(",")(0)).collect.toSet.toArray
val rows = users.groupBy(_._2).map(l => Vectors.sparse(userset.length, l._2.map(r => (userset.indexOf(r._1),r._3.toDouble)).toSeq))

val mat = new RowMatrix(rows)
/**
columnSimilarities(a)
a = none: using normal algorithm
a = float: using DIMSUM algorithm
*/
val sim = mat.columnSimilarities(0.5)
val entries = sim.entries.map { case MatrixEntry(i, j, u) => ((i, j), u) }
entries.saveAsTextFile("DIMSUM")
