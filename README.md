# Project 4 - Data Lake - Purpose
A music streaming startup, Sparkify, has grown their user base and song database even more and want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

# ETL pipeline
The ETL pipeline should extract songs and user log data from S3, processes it using Spark, and loads the data back into S3 as a set of dimensional tables (parquet).

# Project Files
- dl.cfg - contains AWS access keys to store the output parquet files to S3
- etl.py - The ETL pipeline to transform the JSON files present in s3 to more granular dimensional tables(parquet format). And finally store the output table back to S3. You can invoke it by going to the terminal and use the command 'python etl.py'

# Example queries and results for song play analysis.
- Analyze songs of a given artist - 
  song_plays_schema = StructType([])
  df =  spark.read.json(output_data + "log_output/songplays.parquet", schema=song_plays_schema)
  df.createOrReplaceTempView("songplays")
  resultDf = spark.sql('SELECT * FROM songplays where artist_id = "AR5KOSW1187FB35FF4"')
  resultDf.show(5)

