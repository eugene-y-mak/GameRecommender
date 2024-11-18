import pandas as pd
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName('GameRecommender').getOrCreate()

# Load data into a Spark DataFrame (example)
df = spark.read.csv('user_game_data.csv', header=True, inferSchema=True)

# Preprocess the data, e.g., clean missing values or normalize
df = df.dropna(subset=['rating'])  # Drop rows where rating is missing
df = df.withColumn('rating', df['rating'] / 5.0)  # Normalize ratings to [0,1]

# Convert to Pandas DataFrame for smaller operations
df_pd = df.toPandas()
