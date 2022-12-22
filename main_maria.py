from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.functions import mean
import pyspark.sql.dataframe


# import org.apache.spark.sql.SQLContext

#input_dataset_path = "./resources/2000.csv"
input_dataset_path = "C:/Users/USUARIO/Desktop/Master HMDA/1 Semester/BIG DATA/Flight Detection/resources/2000.csv"



# The application will start here as it is executed, we can create a little console menu like in the example
if __name__ == '__main__':


    spark = SparkSession.builder.appName("Linear Regression").master("local[*]").getOrCreate()
    data = spark.read.csv(path=input_dataset_path, inferSchema=True, header=True)
  
    print('----------------PREPROCESSING--------------------------')
    #Eliminating cancelled flights 
    data = data[data.Cancelled != 1]

    # Forbidden variables and useless variables divided by an intro(remember to add CRSElapsedTime and TaxiOut, not added for safety)
    data = data.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
              "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",

              "UniqueCarrier", "FlightNum", "TailNum", "Distance", "Origin", "Dest", "Cancelled", "Cancellationcode", "Year", "Month", "DayofMonth", "DayOfWeek", "CRSElapsedTime")

    #Other columns to drop because of linear dependencies: DepDelay = DepTime - CRSDepTime 
    data = data.drop('DepTime', 'CRSDepTime')

    #Changing numerical datatypes to double  --> should we generalize and do this for all columns?  
    data = data.withColumn('DepDelay', data.DepDelay.cast('double'))
    data = data.withColumn('CRSArrTime', data.CRSArrTime.cast('double'))
    data = data.withColumn('ArrDelay', data.ArrDelay.cast('double'))
    data = data.withColumn('TaxiOut', data.TaxiOut.cast('double'))
    
    #Droping the rows with missing values in the target variable 
    data = data.na.drop(subset=["ArrDelay"])

    #Filling in missing values with 0 in the selected feature columns
    data = data.na.fill(value = 0)  
    data.show(5,False)


    data.printSchema()
    data.show(10, False)
    

    print('-----------------BUILDING MODEL----------------')
    #Prepare independent variable(feature) and dependant variable using assembler
    inputCols = ['DepDelay', 'TaxiOut', 'CRSArrTime']

    vector_assembler = VectorAssembler(inputCols=inputCols, outputCol='features')  
    #Nacho: setHandlerInvalid para eliminar los nulls de DepDelay
    input_dataset_va_df = vector_assembler.setHandleInvalid("skip").transform(data)
    input_dataset_va_fl_df = input_dataset_va_df.select(['features', 'ArrDelay'])
    input_dataset_va_fl_df.show(10, False)

    # Splitting training and testing dataset 70% for training and 30% for testing
    train_test_dataset = input_dataset_va_fl_df.randomSplit([0.7, 0.3], seed=10)  # seed guarantees randomness
    print(type(train_test_dataset))
    train_dataset_df = train_test_dataset[0]
    test_dataset_df = train_test_dataset[1]

    # Training the model
    linear_regression_model = LinearRegression(featuresCol='features',
                                               labelCol='ArrDelay',
                                               maxIter=100,
                                               regParam=0.2,
                                               elasticNetParam=0.8)
    # Building the model
    linear_regression_model = linear_regression_model.fit(train_dataset_df)
    
    # Testing the model
    predictions_df = linear_regression_model.transform(test_dataset_df)
    # The column predictions is the expected result and it is generated by the past functions
    predictions_df.select("prediction", "ArrDelay", "features").show(5, False)

    print('------------------MODEL EVALUATION-----------------')
    # And finally, the evaluation of the model
    model_training_summary = linear_regression_model.summary
    print("RMSE %f" % model_training_summary.rootMeanSquaredError)
    print("r2: %f" % model_training_summary.r2)

    spark.stop()
    print("Demo Program Completed")
