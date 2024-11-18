from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Load the user-game rating data into a Spark DataFrame
ratings = spark.read.csv('ratings.csv', header=True, inferSchema=True)

# Split data into training and test sets
train, test = ratings.randomSplit([0.8, 0.2])

# Initialize ALS model
als = ALS(userCol='user_id', itemCol='game_id', ratingCol='rating', nonnegative=True, implicitPrefs=False)

# Train the ALS model
model = als.fit(train)

# Make predictions
predictions = model.transform(test)

# Evaluate model using RMSE
evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")
