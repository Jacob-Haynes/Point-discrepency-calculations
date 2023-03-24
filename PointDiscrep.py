# Databricks notebook source
from pyspark.sql.functions import col, collect_list, explode, sum, first, row_number, desc, dense_rank, avg, count, min, max, abs, asc
from pyspark.sql.window import Window

# COMMAND ----------

final_votes = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/shared_uploads/jacob.haynes@monterosa.co/final_votes_2023_03_01_max.csv")
#     .load("dbfs:/FileStore/shared_uploads/jacob.haynes@monterosa.co/final_votes_2023_03_01_min.csv")

correct_answers = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/shared_uploads/jacob.haynes@monterosa.co/filteredListWithPoints.csv")

ct_leaderboard = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/shared_uploads/jacob.haynes@monterosa.co/leaderboard__4_.csv")

overall_leaderboard = spark.read \
    .format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("dbfs:/FileStore/shared_uploads/jacob.haynes@monterosa.co/overallleaderboard.csv")

# COMMAND ----------

# increase the integer value of enmasse option by one as the gamify correct option starts from 1 where as enmasse starts from 0
final_votes_optionfix = final_votes.withColumn("option", col("option") + 1)

# Join the final_votes and correct_answers tables on the poll_id and enmasseId columns
merged_table = final_votes_optionfix.join(correct_answers, final_votes_optionfix.poll_id == correct_answers.enmasseId)

# Filter the merged table to keep only the rows where the option matches the correctOption
correct_table = merged_table.filter(col('option') == col('correctOption'))

# Group the correct_table by poll_id and create a new column with a list of the session_ids that got the answer correct
grouped_table = correct_table.groupBy('poll_id') \
    .agg(collect_list('session_id').alias('correct_users'), first('pointsForCorrectAnswer').alias('points_per_poll'))

# Show the final table
grouped_table.show()

# COMMAND ----------

# calculate points for each user
total_points = grouped_table.select("poll_id", explode("correct_users").alias("session_ID"), "points_per_poll") \
    .groupBy("session_ID") \
    .agg(sum("points_per_poll").alias("total_points"))

# Sort the total_points table by points in descending order
total_points = total_points.orderBy("total_points", ascending=False)

# Show the final table
total_points.show()

# COMMAND ----------

# Join the total_points and ct_leaderboard tables on the userId and session_ID columns
merged_table = total_points.join(ct_leaderboard, total_points.session_ID == ct_leaderboard.userId)

# Select the columns needed for the score_comparison table
score_comparison = merged_table.select(
    total_points.session_ID,
    total_points['total_points'].alias('corrected_points'),
    ct_leaderboard['score'].alias('ct_score'),
    ct_leaderboard['publicProfile/username'].alias('publicProfile_username')
)

# Sort the total_points table by points in descending order
score_comparison = score_comparison.orderBy("corrected_points", ascending=False)

# Show the score_comparison table
score_comparison.show()

# COMMAND ----------

corrected_score = score_comparison.groupBy('session_ID', 'publicProfile_username') \
                          .agg(sum('corrected_points').alias('sum_corrected_points'),
                               sum('ct_score').alias('sum_ct_score')) \
                          .withColumn('total_corrected_score', col('sum_corrected_points') + col('sum_ct_score')) \
                          .select('session_ID', 'total_corrected_score', 'publicProfile_username')

# Sort table by points in descending order
corrected_score = corrected_score.orderBy("total_corrected_score", ascending=False)

# Display the output table
corrected_score.show()

# COMMAND ----------

# Join the corrected_score and overall_leaderboard tables on session_id and username
results_joined_table = corrected_score.join(
    overall_leaderboard,
    (corrected_score.session_ID == overall_leaderboard.userId)
)

# Select the required columns
combined_result_table = results_joined_table.select(
    col("session_ID"),
    col("publicProfile_username"),
    col("total_corrected_score"),
    col("score"),
    col("rank"),
)

# Sort table by points in descending order
combined_result_table = combined_result_table.orderBy("total_corrected_score", ascending=False)

# Show the final table
combined_result_table.show()

# COMMAND ----------

window = Window.orderBy(desc("total_corrected_score"))

corrected_rank_column = dense_rank().over(window)

corrected_rank_table = combined_result_table.withColumn("corrected_rank", corrected_rank_column)

difference_table = corrected_rank_table.withColumn("rank_difference", col("rank") - col("corrected_rank"))

difference_table.show()

# COMMAND ----------

# Calculate average rank difference and count of non-zero rank differences
rank_diff_stats = (difference_table
                   .select(avg(abs('rank_difference')).alias('avg_rank_diff'),
                           count('*').alias('count_nonzero_rank_diff'))
                   .first())

# Calculate maximum individual rank difference
max_rank_diff = difference_table.selectExpr('max(abs(rank_difference)) as max_rank_diff').first().max_rank_diff

# Calculate score difference stats
score_diff_stats = (difference_table
                    .select(abs(col('total_corrected_score') - col('score')).alias('score_diff'))
                    .agg(avg('score_diff').alias('avg_score_diff'),
                         max('score_diff').alias('max_score_diff'),
                         min('score_diff').alias('min_score_diff'))
                    .first())

# Print results
print(f"Average rank difference: {rank_diff_stats.avg_rank_diff:.2f}")
print(f"Number of rows with non-zero rank difference: {rank_diff_stats.count_nonzero_rank_diff}")
print(f"Maximum individual rank difference: {max_rank_diff}")
print(f"Average score difference: {score_diff_stats.avg_score_diff:.2f}")
print(f"Maximum score difference: {score_diff_stats.max_score_diff}")
print(f"Minimum score difference: {score_diff_stats.min_score_diff}")


# COMMAND ----------

difference_table.write.format("csv").mode("overwrite").option("header", "true").option("delimiter", ",").option("encoding", "UTF-8").save("dbfs:/FileStore/formulae/difference_table.csv")

# COMMAND ----------

