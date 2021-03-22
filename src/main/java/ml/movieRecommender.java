package ml;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.*;
import scala.collection.JavaConversions;
import scala.collection.mutable.WrappedArray;

import java.util.*;

public class movieRecommender {

    public static void main(String[] args) {
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder().appName("movieRecommender").master("local[*]")
                .config("spark.sql.warehouse.dir", "file:///d:spark_tmp").getOrCreate();

        Dataset<Row> dataset = spark.read().option("header", true)
                .option("inferSchema", true).csv("src/main/resources/ml-100k/ratings.csv");

        int userId = 611;
        int moviesConsidered = 30;
        double avgRating = 3.0;

        dataset = dataset.drop("timestamp");
        dataset.createOrReplaceTempView("ratings");
        spark.sql("select avg(rating) from ratings where userId=" + userId).show();

        // matrix factorisation
        ALS als = new ALS();
        als.setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("movieId")
                .setRatingCol("rating");
        ALSModel model = als.fit(dataset);


        Dataset<Row> newUserList = spark.read().option("inferSchema",true).option("header",true).csv("src/main/resources/newUserList.csv");
        Dataset<Row> userRecs = model.recommendForUserSubset(newUserList,moviesConsidered);

        Dataset<Row> movieData = spark.read().option("inferSchema",true).option("header",true).csv("src/main/resources/ml-100k/movies.csv");
        movieData.createOrReplaceTempView("movies");

        // Getting recommendations in recs list
        List<Row> userRecsList = userRecs.takeAsList(moviesConsidered);
        List<Integer> recs = new ArrayList<>();
        for(Row r: userRecsList)
        {
            userId = r.getAs(0);
            int rec;
            Collection recCollection = JavaConversions.asJavaCollection(((WrappedArray) r.getAs("recommendations")).toList());;
            for (Row row : (Iterable<Row>) recCollection) {
                rec = (int) row.get(0);
                recs.add(rec);
            }
        }

        // Getting user movies and rating them according to their genre
        Dataset<Row> userMovies = spark.sql("select * from ratings where userId=" + userId);
        userMovies = userMovies.join(movieData,"movieId");
        userMovies.show(false);
        Map<String, Double> genreRating = new HashMap<String, Double>();
        List<Row> userMoviesList = userMovies.takeAsList(5);
        for(Row r: userMoviesList)
        {
            double rating = ((double) r.get(2)) - avgRating;
            String[] genres = r.get(4).toString().split("\\|");
            for(String genre: genres)
            {
                if(!genreRating.containsKey(genre))
                {
                    genreRating.put(genre, rating);
                }
                else
                {
                    double prevRating = genreRating.get(genre);
                    genreRating.put(genre,prevRating+rating);
                }
            }
        }

        System.out.println("Genre Rating:");
        for (String i : genreRating.keySet()) {
            System.out.println("key: " + i + " value: " + genreRating.get(i));
        }

        // collecting the recommended movies info from movieData
        Dataset<Row> recMovies, temp;
        recMovies = spark.sql("select * from movies where movieId=" + recs.get(0));
        for(int i=1;i<recs.size();i++)
        {
            int rec = recs.get(i);
            temp = spark.sql("select * from movies where movieId=" + rec);
            recMovies = recMovies.union(temp);
        }

        //scoring the movies from the collaborative filtering result(recs) using genreRating
        List<Row> recMoviesList = recMovies.takeAsList(moviesConsidered);
        Map<String, Double> scores = new HashMap<>();
        List<Pair> scoring = new ArrayList<>();
        for(Row r: recMoviesList)
        {
            double score = 0;
            String movie = r.get(1).toString();
            String[] genres = r.get(2).toString().split("\\|");
            for(String genre: genres)
            {
                if(genreRating.containsKey(genre))
                {
                    score += genreRating.get(genre);
                }
            }
            scores.put(movie,score);
            scoring.add(new Pair(score, movie));
        }
        Collections.sort(scoring, (p1, p2) -> (int) (p2.score - p1.score));

        System.out.println("\nUserId: " + userId + "\nRecommendations:");
        for(Pair p: scoring)
        {
            System.out.println(p.movie + ": " + p.score);
        }
    }
}

class Pair {
    double score;
    String movie;

    public Pair(double score, String movie)
    {
        this.score = score;
        this.movie = movie;
    }
}
