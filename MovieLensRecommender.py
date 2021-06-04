import sys
import pandas as pd
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# add genre columns to movies df
def add_genres(moviesDf, genres):
    for genre in genres:
        moviesDf[genre] = [0] * moviesDf.shape[0]
    return moviesDf


def set_genres(moviesDf):
    for index in moviesDf.index:
        for genre in moviesDf.loc[index].genres.split('|'):
            moviesDf.loc[index, genre] = 1
    moviesDf.drop(columns='genres', inplace=True)
    return moviesDf


def km_model(k, data):
    km = KMeans().setK(k).setSeed(823)
    model = km.fit(data)
    return model


def compute_cost(centers, point):
    cluster = point.cluster
    center = centers.iloc[cluster].to_list()
    coordinates = point.features
    return sum((x1 - x2) ** 2 for (x1, x2) in zip(coordinates, center))


def get_cv_ouptut(cvModel):
    CVModelParams = pd.DataFrame(cvModel.extractParamMap()[cvModel.getParam('estimatorParamMaps')])
    CVModelParams.columns = ['Rank', 'MaxIter', 'RegParam']
    CVModelParams['Training RMSE'] = cvModel.avgMetrics
    return CVModelParams


def kmeans_results(resultsLists):
    resultsDF = pd.DataFrame(resultsLists, columns=['k', 'SSE', 'Training RMSE'])
    resultsDF['SSEDiff'] = resultsDF['SSE'].diff().fillna(value=0)
    resultsDF = resultsDF.reindex(columns=['k', 'SSE', 'SSEDiff', 'Training RMSE'])
    return resultsDF


def print_kmeans_results(bestKMeansModel):
    print("\n**Best K-Means Model**")
    print("(where k corresponds to the largest drop in SSD)\n")
    print("k: {}".format(bestKMeansModel['k']))
    print("SSE: {}".format(bestKMeansModel['SSE']))
    print("Training RMSE = {}".format(bestKMeansModel['Training RMSE']))


def print_als_results(paramsDF):
    bestModel = paramsDF.iloc[alsCVModelParams['Training RMSE'].idxmin()]
    print("\n**Best ALS Model**")
    print("(after using cross-validation with number of folds = 3)\n")
    print("Rank: {}".format(bestModel.Rank))
    print("MaxIter: {}".format(bestModel.MaxIter))
    print("RegParam: {}".format(bestModel.RegParam))
    print("Training RMSE = {}".format(bestModel['Training RMSE']))


def get_avg_ratings(ratingsRDD, kMeansClustersRDD):
    return ratingsRDD \
        .join(kMeansClustersRDD.select('movieId', 'cluster'), on=['movieId'], how='inner') \
        .groupby('userId', 'cluster') \
        .avg("rating") \
        .withColumnRenamed('avg(rating)', 'prediction')


def get_kmeans_predictions(ratingsRDD, kMeansClustersRDD, userClusterRDD):
    return ratingsRDD \
        .join(kMeansClustersRDD.select('movieId', 'cluster'), on=['movieId'], how='inner') \
        .join(userClusterRDD, on=["userId", "cluster"], how="inner")


try:
    directory = sys.argv[1]
except:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          "IMPORTANT NOTE\n"
          "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          "\n\n"
          "To run this code, please use the following template:\n\n"
          "spark-submit MovieLensRecommender.py <directory> \n\n"
          "For the <directory> parameter, please provide the full path of the directory containing the files"
          " movies.dat & ratings.dat. "
          "If you are running this script from the same directory containing these files, then use a ./ instead.\n"
          "The following example shows how the code would be executed if the files are in a directory called "
          "'/home/medium':\n\n"
          "spark-submit MovieLensRecommender.py /home/medium/ \n"
          "\n"
          "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          "END\n"
          "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
          )
    sys.exit(2)

# MAIN
sc = SparkContext('local')
spark = SparkSession(sc)

# Load data
moviesFile = directory + "movies.dat"
movies = sc.textFile(moviesFile)

# create list of genres
genres = list(
    set(movies.map(lambda line: line.split("::"))
        .map(lambda obs: obs[2])
        .flatMap(lambda genre: genre.split("|"))
        .collect()))

moviesDf = pd.DataFrame(movies.map(lambda line: line.split("::")).collect())
moviesDf.columns = ['movieId', 'title', 'genres']
moviesDf = add_genres(moviesDf, genres)
moviesDf = set_genres(moviesDf)

va = VectorAssembler().setInputCols(genres).setOutputCol("features")
movies = va.transform(spark.createDataFrame(moviesDf))

ratingsFile = directory + "ratings.dat"
ratings = spark.read.text(ratingsFile) \
    .rdd.toDF() \
    .selectExpr("split(value , '::') as col") \
    .selectExpr(
    "cast(col[0] as int) as userId",
    "cast(col[1] as int) as movieId",
    "cast(col[2] as float) as rating",
    "cast(col[3] as long) as timestamp") \
    .drop("timestamp")

# Split the data
# ratingsTrain: is comprised of movies the user had already seen and rated
# ratingsTest: is comprised of movies the user hadn't seen. The ratings in this value will be used for evaluation
ratingsTrain, ratingsTest = ratings.randomSplit([0.7, 0.3], seed=823)

# Create RMSE evaluator
evaluator = RegressionEvaluator() \
    .setMetricName("rmse") \
    .setLabelCol("rating") \
    .setPredictionCol("prediction")

# KMeans
# Find best k using training data set
k = range(2, 11)
kMeansResults = []
for i in k:
    model = km_model(i, movies)
    centers = pd.DataFrame(model.clusterCenters())
    kMeansClusters = model.transform(movies)
    kMeansClusters = kMeansClusters.withColumnRenamed('prediction', 'cluster')

    # Calculate SSE
    SSE = kMeansClusters.rdd.map(lambda point: compute_cost(centers, point)).sum()

    # Get user average rating per cluster using ratingsTrain data set
    userClusterRating = get_avg_ratings(ratingsTrain, kMeansClusters)

    # Make predictions using users' average cluster ratings
    # Using training data set
    kMeansPredictionsTraing = get_kmeans_predictions(ratingsTrain, kMeansClusters, userClusterRating)

    # Calculate RMSE
    RMSETrain = evaluator.evaluate(kMeansPredictionsTraing)

    # Tabulate the results
    kMeansResults.append([i, SSE, RMSETrain])


# Identify best K according to SSE differential
kMeansResultsDF = kmeans_results(kMeansResults)
bestKMeansModel = kMeansResultsDF.iloc[kMeansResultsDF['SSEDiff'].idxmin()]

# Use best k to build model for test data set
model = km_model(bestKMeansModel['k'], movies)
kMeansClusters = model.transform(movies)
kMeansClusters = kMeansClusters.withColumnRenamed('prediction', 'cluster')

# Get user average rating per cluster using ratingsTrain data set
userClusterRating = get_avg_ratings(ratingsTrain, kMeansClusters)

# Make predictions using users' average cluster ratings
# Using training data set
kMeansPredictionsTest = get_kmeans_predictions(ratingsTest, kMeansClusters, userClusterRating)

# Calculate RMSE
RMSETest = evaluator.evaluate(kMeansPredictionsTest)

################################################################

# ALS
# Create ALS model
als = ALS() \
    .setSeed(823) \
    .setUserCol("userId") \
    .setItemCol("movieId") \
    .setRatingCol("rating") \
    .setImplicitPrefs(False) \
    .setColdStartStrategy("drop")

# Define grid parameters
ranks = [1, 10, 100]
maxIters = [5, 10, 15]
lambdas = [0.001, 0.01, 0.05]

# Create parameter grid
paramMap = ParamGridBuilder() \
    .addGrid(als.rank, ranks) \
    .addGrid(als.maxIter, maxIters) \
    .addGrid(als.regParam, lambdas) \
    .build()

# Run cross-validation, and choose the best set of parameters.
alsCV = CrossValidator(estimator=als,
                       estimatorParamMaps=paramMap,
                       evaluator=evaluator,
                       numFolds=3)

# Run the cv on the training data
alsCVModel = alsCV.fit(ratingsTrain)
alsCVModelParams = get_cv_ouptut(alsCVModel)

# Make predictions on test documents. cvModel uses the best model found.
alsBestModel = alsCVModel.bestModel
alsPredictions = alsBestModel.transform(ratingsTest)

# Evaluate the model by computing the RMSE on the test data
alsRMSETest = evaluator.evaluate(alsPredictions)

################################################################

# Print results
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("Results Report\n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("KMeans Results\n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print(kMeansResultsDF, "\n")
print_kmeans_results(bestKMeansModel)
print("Using the best KMeans model, the Test RMSE is: {}".format(RMSETest))
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("ALS Results\n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print(alsCVModelParams)
print_als_results(alsCVModelParams)
print("Using the best ALS model, the Test RMSE is {}".format(alsRMSETest))
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
print("End Report\n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
