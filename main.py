from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,VectorIndexer
from pyspark.ml.regression import LinearRegression, RandomForestRegressor,IsotonicRegression
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark import SparkConf
import pyspark.sql.dataframe
from pyspark.sql.functions import mean
from pyspark.ml.evaluation import RegressionEvaluator

import pyspark.sql.dataframe
from tkinter import *
from tkinter import ttk, font
import getpass
import pandas as pd

# import org.apache.spark.sql.SQLContext

# input_dataset_path = "./resources/2000.csv"
#  C:/Users/eleni/PycharmProjects/bigData/resources/2000.csv
input_dataset_path = ""

class App():
    def __init__(self):
        self.root = Tk()
        self.root.title("Flight Delay Predictor")
        self.root.geometry('900x700')
        self.root.iconbitmap('icon.ico')
        fbold = font.Font(weight='bold')
        self.tag1 = ttk.Label(self.root, text="Please introduce the path to your dataset:",
                               font=fbold)
        self.path = StringVar()
        self.path.set("")
        print('path')
        print(type(self.path))
        print(self.path.get())
        print(self.path)
        self.ctext1 = ttk.Entry(self.root,
                                textvariable=self.path)
        self.separ1 = ttk.Separator(self.root, orient=HORIZONTAL)
        #buttons in the window
        #button to call the predictor with the path of the dataset
        self.button1 = ttk.Button(self.root, text="Predict",
                                 command=self.predict)
        #button to close the app
        self.button2 = ttk.Button(self.root, text="Cancel",
                                 command=quit)
        #positions of the widgets
        self.tag1.pack(side=TOP, fill=BOTH, expand=True,
                        padx=5, pady=5)
        self.ctext1.pack(side=TOP, fill=X, expand=True,
                         padx=5, pady=5)

        self.separ1.pack(side=TOP, fill=BOTH, expand=True,
                         padx=5, pady=10)
        self.button2.pack(side=BOTTOM, fill=X, expand=True,
                          padx=5, pady=5)
        self.button1.pack(side=BOTTOM, fill=X, expand=True,
                         padx=5, pady=5)

        #focus on the path text field

        self.tinfo = Text(self.root, width=100, height=20)
        self.tinfo.configure(state="disabled")
        scroll_bar = Scrollbar(self.root)
        scroll_bar.pack(side=RIGHT)
        self.tinfo.pack(side=TOP)

        self.ctext1.focus_set()
        self.root.mainloop()
    def predict(self):
        if pd.notnull(self.path.get()):
            texto_info =""
            print('PATH')
            print(self.path.get())#this is VERY MUCH OK
            #call App2 which is a popup with the results and a button to call app 1 again
            #remember to clear all of the global variables!!!
            self.tinfo.delete("1.0", END)
            #texto_info = "Clase de 'raiz': " + info1 + "\n"
            #texto_info += "Resolución y posición: " + info2 + "\n"


            spark = SparkSession.builder.appName("Linear Regression").master("local[*]").getOrCreate()
            data = spark.read.csv(path=self.path.get(), inferSchema=True, header=True)

            print('----------------PREPROCESSING--------------------------')
            # Eliminating cancelled flights
            data = data[data.Cancelled != 1]
            texto_info += "-----PREPROCESSING-----" + "\n" + "\n"
            self.tinfo.configure(state='normal')
            self.tinfo.insert(END, texto_info)
            self.tinfo.configure(state='disabled')
            texto_info = ""

            # Forbidden variables and useless variables divided by an intro(remember to add CRSElapsedTime and TaxiOut, not added for safety)
            data = data.drop("ArrTime", "ActualElapsedTime", "AirTime", "TaxiIn", "Diverted", "CarrierDelay",
                             "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay",

              "UniqueCarrier", "FlightNum", "TailNum", "Origin", "Dest", "Cancelled", "Cancellationcode", "Year", "Month", "DayofMonth", "DayOfWeek", "CRSElapsedTime")

            # Other columns to drop because of linear dependencies: DepDelay = DepTime - CRSDepTime
            data = data.drop('DepTime', 'CRSDepTime')

            #Changing numerical datatypes to double  --> should we generalize and do this for all columns?
            data = data.withColumn('DepDelay', data.DepDelay.cast('double'))
            data = data.withColumn('CRSArrTime', data.CRSArrTime.cast('double'))
            data = data.withColumn('ArrDelay', data.ArrDelay.cast('double'))
            data = data.withColumn('TaxiOut', data.TaxiOut.cast('double'))
            data = data.withColumn('Distance', data.TaxiOut.cast('double'))

            #Droping the rows with missing values in the target variable
            data = data.na.drop(subset=["ArrDelay"])

            #Filling in missing values with 0 in the selected feature columns
            data = data.na.fill(value = 0)
            data.show(5,False)


            data.printSchema()
            data.show(10, False)


            print('-----------------BUILDING MODEL----------------')
            #Prepare independent variable(feature) and dependant variable using assembler
            inputCols = ['DepDelay', 'TaxiOut', 'CRSArrTime', 'Distance']
            print('-----------------BUILDING MODEL----------------')
            texto_info += "-----BUILDING MODEL-----" + "\n"
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
            # Building the vectorIndexer for decision tree
            featureIndexer = VectorIndexer(inputCol="features",
                                           outputCol="indexedFeatures",
                                           maxCategories=300).fit(input_dataset_va_fl_df)

            # Building the regressions models
            linear_regression_model = LinearRegression(featuresCol='features',
                                                       labelCol='ArrDelay',
                                                       maxIter=100,
                                                       regParam=0.2,
                                                       elasticNetParam=0.8)
            dt = RandomForestRegressor(featuresCol="indexedFeatures", labelCol='ArrDelay')

            Random_Forest_model = RandomForestRegressor(featuresCol='features',
                                                        labelCol='ArrDelay')
            Isotonic_regression_model = IsotonicRegression(featuresCol='features',
                                                           labelCol='ArrDelay')

            # fitting the model
            linear_regression_model = linear_regression_model.fit(train_dataset_df)
            pipeline = Pipeline(stages=[featureIndexer, dt])
            model = pipeline.fit(train_dataset_df)

            Random_Forest_model = Random_Forest_model.fit(train_dataset_df)

            Isotonic_regression_model = Isotonic_regression_model.fit(train_dataset_df)
            # Testing the model
            predictions_df = linear_regression_model.transform(test_dataset_df)
            Random_Forest_model_predictions = Random_Forest_model.transform(test_dataset_df)

            predictions_rf = model.transform(test_dataset_df)
            predictions_ir = Isotonic_regression_model.transform(test_dataset_df)
            # The column predictions is the expected result and it is generated by the past functions
            predictions_df.select("prediction", "ArrDelay", "features").show(5, False)

            predictions_pandas = pd.DataFrame(predictions_df.head(5), columns=["features","ArrDelay","prediction"])
            #predictions_reduced_df = predictions_df.select("prediction", "ArrDelay", "features")
            #texto_info += predictions_pandas.to_string()
            self.tinfo.configure(state='normal')
            self.tinfo.insert(END, texto_info)
            self.tinfo.configure(state='disabled')
            texto_info = ""
            print(predictions_pandas.to_string())

            #texto_info += "\n"
            #texto_info += "\n"
            texto_info += "\n"

            print('-----MODEL EVALUATION-----')
            # And finally, the evaluation of the model
            model_training_summary = linear_regression_model.summary

            evaluator = RegressionEvaluator()
            evaluator.setPredictionCol("prediction")
            evaluator.setLabelCol("ArrDelay")
            r2_dt = evaluator.evaluate(predictions_rf, {evaluator.metricName: "r2"})
            rmse_dt = evaluator.evaluate(predictions_rf, {evaluator.metricName: "rmse"})

            r2 = evaluator.evaluate(Random_Forest_model_predictions, {evaluator.metricName: "r2"})
            rmse = evaluator.evaluate(Random_Forest_model_predictions, {evaluator.metricName: "rmse"})

            r2_ir = evaluator.evaluate(predictions_ir, {evaluator.metricName: "r2"})
            rmse_ir = evaluator.evaluate(predictions_ir, {evaluator.metricName: "rmse"})

            texto_info += "-----MODEL EVALUATION-----" + "\n"
            texto_info += "linear regression" + "\n"
            texto_info += "RMSE " + str(model_training_summary.rootMeanSquaredError) + "\n"
            texto_info += "r2: " + str(model_training_summary.r2) + "\n"
            texto_info += "\n"
            texto_info += "decision tree" + "\n"
            texto_info += "RMSE " + str(rmse_dt) + "\n"
            texto_info += "r2: " + str(r2_dt) + "\n"
            texto_info += "\n"
            texto_info += "random forest" + "\n"
            texto_info += "RMSE " + str(rmse) + "\n"
            texto_info += "r2: " + str(r2) + "\n"
            texto_info += "\n"
            texto_info += "Isotonic regression" + "\n"
            texto_info += "RMSE " + str(rmse_ir) + "\n"
            texto_info += "r2: " + str(r2_ir) + "\n"
            texto_info += "\n"

            print("linear regression")
            print("RMSE %f" % model_training_summary.rootMeanSquaredError)
            print("r2: %f" % model_training_summary.r2)

            print("decision tree")
            print("RMSE %f" % rmse_dt)
            print("r2: %f" % r2_dt)

            print("random forest")
            print("RMSE %f" % rmse)
            print("r2: %f" % r2)

            print(" Isotonic_regression")
            print("RMSE %f" % rmse_ir)
            print("r2: %f" % r2_ir)

            spark.stop()
            print("Demo Program Completed")
            self.tinfo.configure(state='normal')
            self.tinfo.insert(END, texto_info)
            self.tinfo.configure(state='disabled')
            texto_info = ""
            #destroy de window we used to input the path
            #self.root.destroy()
            #App2()
        else:
            print("Acceso denegado")

            # Se inicializa la variable 'self.clave' para
            # que el widget 'self.ctext2' quede limpio.
            # Por último, se vuelve a asignar el foco
            # a este widget para poder escribir una nueva
            # contraseña.

            self.path.set("")


# The application will start here as it is executed, we can create a little console menu like in the example
if __name__ == '__main__':

    print("Hello")
    mi_app = App()


