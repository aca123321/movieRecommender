package ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.*;
import scala.collection.JavaConversions;
import scala.collection.mutable.WrappedArray;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public class movieRecommender {

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("Gym Competitors").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///d:spark_tmp").getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true)
                .option("inferSchema", true).csv("src/main/resources/ml-100k/ratings.csv");

        dataset = dataset.drop("timestamp");

//        dataset = dataset.withColumn("rating",col("proportionWatched").multiply(100)).drop("proportionWatched");
//        dataset = dataset.groupBy("userId").pivot("courseId").sum("rating");
        dataset.describe();

        ALS als = new ALS();
        als.setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating");

        ALSModel model = als.fit(dataset);

        Dataset<Row> newUserList = spark.read().option("inferSchema",true).option("header",true).csv("src/main/resources/newUserList.csv");
        Dataset<Row> userRecs = model.recommendForUserSubset(newUserList,5);

        Dataset<Row> movieData = spark.read().option("inferSchema",true).option("header",true).csv("src/main/resources/ml-100k/movies.csv");
        movieData.createOrReplaceTempView("movies");

        List<Row> userRecsList = userRecs.takeAsList(5);
        List<Integer> recs = new ArrayList<>();
        for(Row r: userRecsList)
        {
            int userId = r.getAs(0);
            int rec;
            Collection javaCollection = JavaConversions.asJavaCollection(((WrappedArray) r.getAs("recommendations")).toList());;
            for (Row row : (Iterable<Row>) javaCollection) {
                rec = (int) row.get(0);
                recs.add(rec);
            }
            System.out.println("User: " + userId);
        }

        System.out.println("Recommendations:");
        Dataset<Row> recMovies, temp;
        recMovies = spark.sql("select * from movies where movieId=" + recs.get(0));
        for(int i=1;i<recs.size();i++)
        {
            int rec = recs.get(i);
            temp = spark.sql("select * from movies where movieId=" + rec);
            System.out.println(rec);
            recMovies = recMovies.union(temp);
        }

        System.out.println("Recommended Movies:");
        recMovies.show(false);
    }
}
