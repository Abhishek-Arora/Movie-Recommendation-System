from pyspark import SparkConf,SparkContext,SQLContext
import sys,unicodedata
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import levenshtein
from pyspark.mllib.recommendation import ALS


def trainALSmodel(trainingSet, moviesNotRatedByUser):

	rank = 5
	numIterations = 12
	model = ALS.train(trainingSet,rank,numIterations)
	predictions = model.predictAll(moviesNotRatedByUser).map(lambda prediction: (prediction.product,prediction.rating))

	return(predictions)
	

## This method tries to match the user input ratings based on levenshtein distance	
def matchUserRatings(moviesDF,userProvidedRatingsDF):

	allRatings = userProvidedRatingsDF.join(moviesDF).select('*',levenshtein(userProvidedRatingsDF.user_title,moviesDF.title).alias('distance')).cache()
	bestMatch = allRatings.groupBy('user_title').agg({'distance':'min'}).withColumnRenamed('min(distance)','min_dis').cache()

	join_condition = [allRatings.user_title == bestMatch.user_title, allRatings.distance == bestMatch.min_dis]

	joinBestMatchWithAllRatings = bestMatch.join(allRatings,join_condition).select('movie_id','user_ratings').withColumnRenamed('user_ratings','Rating')
	matchedUserRatings  = joinBestMatchWithAllRatings.withColumn('user_id',joinBestMatchWithAllRatings.Rating - joinBestMatchWithAllRatings.Rating)

	return(matchedUserRatings)


def main():
	
	conf = SparkConf().setAppName('Movie Recommendation System')
	sc = SparkContext(conf=conf)
	assert sc.version >= '1.5.1'
	sqlContext = SQLContext(sc)

	input = sys.argv[1]
	userRatings = sys.argv[2]
	output = sys.argv[3]

	#Defining schemas to be used for creating dataframes later on

	schemaForMovies = StructType([
	StructField('movie_id', StringType(), False),StructField('title', StringType(), False)
	])

	schemaForRecommendation = StructType([
	StructField('movie_id', StringType(), False),StructField('rating', StringType(), False)
	])

	schemaForUserRatings = StructType([
	StructField('user_ratings', IntegerType(), False),StructField('user_title', StringType(), False)
	])


	## Read ratings and movie data from the input directory

	ratings = sc.textFile(input + "/ratings.dat").map( lambda line: line.split("::")).map(lambda (user_id,movie_id,rating, timestamp): (int(user_id),movie_id,rating)).cache()

	## Movie title should be normalized in order to be able to compare it with the user input ratings.
	movies = sc.textFile(input + "/movies.dat").map(lambda line: line.split("::")).map(lambda (movie_id, title, genre): (movie_id, unicodedata.normalize('NFD',title))).cache()

	userProvidedRatings = sc.textFile(userRatings).map(lambda line: line.split(' ',1)).map(lambda (user_ratings,user_title): (int(user_ratings),unicodedata.normalize('NFD',user_title))).cache()

	moviesDF = sqlContext.createDataFrame(movies,schemaForMovies).cache()
	userProvidedRatingsDF = sqlContext.createDataFrame(userProvidedRatings,schemaForUserRatings).cache()
	matchedUserRatings = matchUserRatings(moviesDF,userProvidedRatingsDF).rdd.map(lambda (movie_id,rating,user_id): (user_id,movie_id,rating))

	## Find out movies that are not rated by the user.
	moviesRatedByUser = matchedUserRatings.map(lambda (user_id,movie_id,rating): movie_id ).collect()
	moviesNotRatedByUser = movies.filter(lambda x: x[0] not in moviesRatedByUser).map(lambda x: (0, x[0]))

	#Create training set on which model will be trained
	trainingSet = ratings.union(matchedUserRatings)

	predictions = trainALSmodel(trainingSet, moviesNotRatedByUser)

	predictionsForUserDF = sqlContext.createDataFrame(predictions,schemaForRecommendation)

	predictionsForUserDF.join(moviesDF,[predictionsForUserDF.movie_id == moviesDF.movie_id]).select(moviesDF.title,predictionsForUserDF.rating).orderBy(predictionsForUserDF.rating.desc()).limit(10).rdd.map(lambda (Title,Ratings): u"Movie: %s Rating: %s" % (Title, Ratings)).saveAsTextFile(output)
	
	
if __name__ == "__main__":
    main()

