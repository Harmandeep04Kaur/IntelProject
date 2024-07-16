# -*- coding: utf-8 -*-
"""Data_Preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HXn_Ig6Fv8T2yvfaccej-DA_tYH4TGZd
"""


class DataPreprocessing:

    def __init__(self) -> None:
        pass

    file_name = ""

    def open_file(self):
        import tkinter as tk
        from tkinter import filedialog
        # Create a root window and hide it immediately
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Set the root window to always be on top
        root.attributes("-topmost", True)

        # Open the file dialog with CSV file filter
        filename = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv")],
            parent=root  # Ensure the dialog is a child of the hidden root window
        )

        # Destroy the root window
        root.destroy()

        return filename

    def readCsv(self, path=None, session=None):

        if path is None:
            path = self.open_file()

        from pyspark.sql import SparkSession  # Module to Create Spark Session
        if session is None:
            session = SparkSession.builder.appName(
                "Data Preprocessing").getOrCreate()

        dataframe = session.read.csv(path, header=True, inferSchema=True)
        path = path.split("/")
        self.file_name = path[-1]
        chars_to_replace = "#@$%^&*()-+=}{[]|\\:;\"'<>,/?. "
        new_columns = dataframe.columns
        translation_table = str.maketrans(
            chars_to_replace, '_' * len(chars_to_replace))
        new_columns = [col.translate(translation_table) for col in new_columns]
        dataframe = dataframe.toDF(*new_columns)
        return dataframe

    """Function to Fill Missing Values"""

    def fillMissingValues(self, dataframe, numeric_cols, categorical_cols):

        # Module to perform various operations on DataFrame
        import pyspark.sql.functions as F

        # Replace missing values with None
        dataframe = dataframe.replace(['?', 'NA', 'Nan', 'na', 'NaN'], None)

        for col in categorical_cols:  # Fill with the most frequent value
            mode_value = dataframe.select(F.mode(dataframe[col])).first()[
                0]  # Calculate the mode value
            # Fill with the mode value
            dataframe = dataframe.fillna(mode_value, subset=[col])

        for col in numeric_cols:  # Fill with the mean value
            mean_value = dataframe.select(F.mean(dataframe[col])).first()[
                0]  # Calculate the mean value
            # Fill with the mean value
            dataframe = dataframe.fillna(mean_value, subset=[col])

        return dataframe

    """Function to Remove Outliers using IQR(Interquartile Range)"""

    def removeOutliers(self, dataframe, cols):
        # Module to perform various operations on DataFrame
        from pyspark.sql.functions import col
        # Calculate the IQR for each column
        for col_name in cols:
            q1 = dataframe.approxQuantile(col_name, [0.25], 0.05)[0]
            q3 = dataframe.approxQuantile(col_name, [0.75], 0.05)[0]
            iqr = q3 - q1
            # Remove outliers using IQR
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            # Filter out the outliers
            dataframe = dataframe.filter((col(col_name) >= lower_bound) &
                                         (col(col_name) <= upper_bound))
        return dataframe

    """Function to Remove Outliers using Z-Score Method"""

    def RemoveOutliers_ZScore(self, dataframe, columns):
        # Module to perform various operations on DataFrame
        from pyspark.sql.functions import col
        for col_name in columns:
            # Calculate mean and standard deviation for the column
            mean_value = dataframe.selectExpr(
                f'mean({col_name}) as mean').collect()[0]['mean']
            stddev_value = dataframe.selectExpr(
                f'stddev({col_name}) as stddev').collect()[0]['stddev']

            # Remove outliers using Z-score
            threshold = 3 * stddev_value
            # Filter out the outliers
            dataframe = dataframe.filter(col(col_name).between(
                mean_value - threshold, mean_value + threshold))

        return dataframe

    """Function to Normalize Numeric Columns using MinMaxScalar"""

    def NormalizeNumericColumns(self, dataframe, cols):
        from pyspark.ml import Pipeline  # Module to create a pipeline of stages
        # Module to perform various feature engineering operations
        from pyspark.ml.feature import VectorAssembler, MinMaxScaler
        # VectorAssembler to Assemble Columns in a Single Column named Features
        assembler = VectorAssembler(
            inputCols=cols, outputCol="UnscaledFeatures")

        # Apply MinMaxScaler to the vector column
        scaler = MinMaxScaler(inputCol="UnscaledFeatures",
                              outputCol="features")

        # Create a pipeline
        pipeline = Pipeline(stages=[assembler, scaler])

        # Fit the pipeline and transform the data
        pipeline_model = pipeline.fit(dataframe)
        dataframe = pipeline_model.transform(dataframe)

        dataframe = dataframe.drop("UnscaledFeatures")

        return dataframe

    """Function to Standardize Numeric Columns using Standard Scaler"""

    def StandardizeNumericColumns(self, dataframe, cols):
        from pyspark.ml import Pipeline  # Module to create a pipeline of stages
        # Module to perform various feature engineering operations
        from pyspark.ml.feature import VectorAssembler, StandardScaler
        # VectorAssembler to Assemble Columns in a Single Column named Features
        assembler = VectorAssembler(
            inputCols=cols, outputCol="UnscaledFeatures")

        # Apply StandardScaler to the vector column
        scaler = StandardScaler(
            inputCol="UnscaledFeatures", outputCol="features", withMean=True, withStd=True)

        # Create a pipeline
        pipeline = Pipeline(stages=[assembler, scaler])

        # Fit the pipeline and transform the data
        pipeline_model = pipeline.fit(dataframe)
        dataframe = pipeline_model.transform(dataframe)

        dataframe = dataframe.drop("UnscaledFeatures")

        return dataframe

    """Function to Encode Categorical Columns using Label Encoding"""

    def LabelEncoding(self, dataframe, categorical_cols):
        from pyspark.ml import Pipeline  # Module to create a pipeline of stages
        # Module to perform various feature engineering operations
        from pyspark.ml.feature import StringIndexer
        # Module to specify the data type of a column
        from pyspark.sql.types import BooleanType
        for column in categorical_cols:
            # Check if the column has only two unique values
            if dataframe.schema[column].dataType == BooleanType():
                dataframe = dataframe.withColumn(
                    column, dataframe[column].cast('int'))
            else:
                # StringIndexer to convert categorical column to numerical indices
                indexer = StringIndexer(
                    inputCol=column, outputCol=f"{column}Index")

                # Create a pipeline
                pipeline = Pipeline(stages=[indexer])

                # Fit the pipeline and transform the data
                pipeline_model = pipeline.fit(dataframe)
                dataframe = pipeline_model.transform(dataframe)

                dataframe = dataframe.withColumn(column, dataframe[f"{column}Index"]).drop(
                    dataframe[f"{column}Index"])

                # indexer_model = pipeline_model.stages[0]  # Get the StringIndexerModel from the pipeline
                # labels = indexer_model.labels  # Get the list of labels
                # print("Category to Index Mapping:")
                # for index, label in enumerate(labels):
                #   print(f"{label}: {index}")

        return dataframe

    def oneHotEncoding(self, dataframe, categorical_cols):
        # Module to perform various feature engineering operations
        from pyspark.ml.feature import StringIndexer, OneHotEncoder
        from pyspark.ml import Pipeline  # Module to create a pipeline of stages
        from pyspark.sql.types import BooleanType
        for column in categorical_cols:
            # Check if the column has only two unique values
            if dataframe.schema[column].dataType == BooleanType():
                dataframe = dataframe.withColumn(
                    column, dataframe[column].cast('int'))
            else:
                # String Indexing
                indexer = StringIndexer(
                    inputCol=column, outputCol=f"{column}Index")

                # Hot Encoding
                encoder = OneHotEncoder(
                    inputCol=f"{column}Index", outputCol=f"{column}Vec")

                # Create a pipeline
                pipeline = Pipeline(stages=[indexer, encoder])

                # Fit the pipeline and transform the data
                pipeline_model = pipeline.fit(dataframe)
                dataframe = pipeline_model.transform(dataframe)

                dataframe = dataframe.withColumn(column, dataframe[f"{column}Vec"]).drop(
                    dataframe[f"{column}Index"], dataframe[f"{column}Vec"])

                # indexer_model = pipeline_model.stages[0]  # Get the StringIndexerModel from the pipeline
                # labels = indexer_model.labels  # Get the list of labels
                # print("Category to Index Mapping:")
                # for index, label in enumerate(labels):
                #   print(f"{label}: {index}")

        return dataframe


if __name__ == "__main__":
    from pyspark.sql import SparkSession  # Module to Create Spark Session
    import matplotlib.pyplot as plt  # Module to plot graphs

    spark = SparkSession.builder.master("local").appName(
        "Data Preprocessing").getOrCreate()
    dp = DataPreprocessing()
    dataframe = dp.readCsv(session=spark)
    target = dataframe.columns[-1]

    if 'id' in dataframe.columns:
        dataframe = dataframe.drop('id')

    dataframe.printSchema()

    distinct_values = dataframe.groupBy(target).count()
    distinct_values.show()

    numeric_cols = []
    categorical_cols = []

    for column in dataframe.dtypes:  # Check the data type of each column
        # If the data type is string, it is categorical
        if column[1] == 'string' or column[1] == 'boolean':
            categorical_cols.append(column[0])
        else:  # If the data type is numeric, it is numeric
            numeric_cols.append(column[0])

    print(dataframe.count())

    dataframe = dp.fillMissingValues(dataframe, numeric_cols, categorical_cols)

    for column in numeric_cols:
        mean_value_before = dataframe.selectExpr(
            f'mean({column}) as mean').collect()[0]['mean']
        stddev_value_before = dataframe.selectExpr(
            f'stddev({column}) as stddev').collect()[0]['stddev']
        # Plotting before outlier removal
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe.select(column).rdd.flatMap(lambda x: x).collect(),
                 bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=mean_value_before, color='red',
                    linestyle='--', label='Mean')
        plt.axvline(x=mean_value_before + 3 * stddev_value_before,
                    color='orange', linestyle='--', label='3 Std Dev')
        plt.axvline(x=mean_value_before - 3 * stddev_value_before,
                    color='orange', linestyle='--')
        plt.title('Distribution Before Outlier Removal')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    columns = numeric_cols[:]
    if target in columns:
        columns.remove(target)

    dataframe = dp.RemoveOutliers_ZScore(dataframe, columns)
    print(dataframe.count())

    for column in numeric_cols:
        mean_value_after = dataframe.selectExpr(
            f'mean({column}) as mean').collect()[0]['mean']
        stddev_value_after = dataframe.selectExpr(
            f'stddev({column}) as stddev').collect()[0]['stddev']
        # Plotting before outlier removal
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe.select(column).rdd.flatMap(lambda x: x).collect(),
                 bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=mean_value_after, color='red',
                    linestyle='--', label='Mean')
        plt.axvline(x=mean_value_after + 3 * stddev_value_after,
                    color='orange', linestyle='--', label='3 Std Dev')
        plt.axvline(x=mean_value_after - 3 * stddev_value_after,
                    color='orange', linestyle='--')
        plt.title('Distribution after Outlier Removal')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    # dataframe = dp.oneHotEncoding(dataframe, categorical_cols)

    dataframe = dp.LabelEncoding(dataframe, categorical_cols)

    dataframe.show()

    columns.extend(categorical_cols)
    if target in columns:
        columns.remove(target)
    dataframe = dp.NormalizeNumericColumns(dataframe, columns)

    dataframe.select("features").show(truncate=False)

    for column in numeric_cols:
        mean_value_after = dataframe.selectExpr(
            f'mean({column}) as mean').collect()[0]['mean']
        stddev_value_after = dataframe.selectExpr(
            f'stddev({column}) as stddev').collect()[0]['stddev']
        # Plotting before outlier removal
        plt.figure(figsize=(10, 6))
        plt.hist(dataframe.select(column).rdd.flatMap(lambda x: x).collect(),
                 bins=20, color='skyblue', edgecolor='black')
        plt.axvline(x=mean_value_after, color='red',
                    linestyle='--', label='Mean')
        plt.axvline(x=mean_value_after + 3 * stddev_value_after,
                    color='orange', linestyle='--', label='3 Std Dev')
        plt.axvline(x=mean_value_after - 3 * stddev_value_after,
                    color='orange', linestyle='--')
        plt.title('Distribution after Outlier Removal')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True)
        plt.show()

    spark.stop()