from flask import Flask, render_template, jsonify, request
from pyspark.sql import SparkSession
from Data_Preprocessing import DataPreprocessing
from Knowledge_Representation import KnowledgeRepresentation
from PatternIdentification_and_InsightGeneration import InsightGeneration

app = Flask(__name__)

session = SparkSession.builder.appName("Dataset Analysis").getOrCreate()
dp = DataPreprocessing()
kr = KnowledgeRepresentation()
ig = InsightGeneration()
df = None
df_pandas = None
target = None
numeric_cols = None
categorical_cols = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/insight.html')
def insight():
    return render_template('insight.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/browse_file', methods=['POST'])
def browse_file():
    global df
    try:
        import os
        file_path = dp.open_file()
        df = dp.readCsv(file_path, session=session)
        file_name = os.path.basename(file_path)
        return jsonify({'result': 'success', 'file_name': file_name})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': 'File Not Uploaded!'})


@app.route('/file_input', methods=['POST'])
def process_input():
    global df
    try:
        import os
        input_data = request.json['input_value']
        if (input_data == '' or input_data.endswith('.csv') == False):
            return jsonify({'result': 'failure', 'error': 'File Name is Invalid!'})

        current_directory = os.getcwd()
        parent_directory = os.path.abspath(
            os.path.join(current_directory, os.pardir))
        path = os.path.join(os.path.join(
            parent_directory, "Data\\"), input_data)

        df = dp.readCsv(path, session)
        return jsonify({'result': 'success'})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': f"File {input_data} Not Found in Data Folder!"})


@app.route('/get_target', methods=['POST'])
def get_target():
    global df
    global target
    target = request.form.get('input_text')
    chars_to_replace = "#@$%^&*()-+=}{[]|\\:;\"'<>,/?. "
    translation_table = str.maketrans(
        chars_to_replace, '_' * len(chars_to_replace))
    target = target.translate(translation_table)
    try:
        if target in df.columns:
            return jsonify({'result': 'success'})
        elif target == '':
            return jsonify({'result': 'failure', 'error': 'Target Column is Blank!'})
        else:
            return jsonify({'result': 'failure', 'error': f"No Target column named {target}"})
    except Exception:
        return jsonify({'result': 'failure', 'error': 'No Dataset Uploaded!'})


@app.route('/fill_missing_values', methods=['POST'])
def additional_processing():
    try:
        global df
        global numeric_cols
        global categorical_cols
        global df_pandas
        numeric_cols = []
        categorical_cols = []
        for column in df.dtypes:  # Check the data type of each column
            # If the data type is string, it is categorical
            if column[1] == 'string' or column[1] == 'boolean':
                categorical_cols.append(column[0])
            else:  # If the data type is numeric, it is numeric
                numeric_cols.append(column[0])

        df = dp.fillMissingValues(df, numeric_cols, categorical_cols)
        df_pandas = df.toPandas()
        return jsonify({'result': 'success'})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/describe_stats', methods=['GET'])
def describe_stats():
    try:
        global df_pandas
        global target
        stats, skewness = kr.show_describe_stats(df_pandas, target)
        return jsonify({'result': 'success', 'stats': stats, 'skew': skewness})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/frequency_plots')
def frequency_plots():
    try:
        global df_pandas
        global target
        plot_data = kr.show_frequencies(df_pandas, target)
        return jsonify({'result': 'success', 'plot': plot_data})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/distribution_plots')
def distribution_plots():
    try:
        global df_pandas
        global target
        plot_data = kr.show_distribution(df_pandas, target)
        return jsonify({'result': 'success', 'plot': plot_data})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/data_refining')
def data_refining():
    try:
        global df
        global target
        global numeric_cols
        global categorical_cols
        columns = numeric_cols[:]
        if target in columns:
            columns.remove(target)
        df = dp.RemoveOutliers_ZScore(df, columns)
        df = dp.LabelEncoding(df, categorical_cols)
        columns.extend(categorical_cols)
        if target in columns:
            columns.remove(target)
        df = dp.NormalizeNumericColumns(df, columns)
        return jsonify({'result': 'success'})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/kmeans_insights')
def kmeans_insights():
    try:
        global df
        insights, plot_data = ig.KMeansClustering(df)
        return jsonify({'result': 'success', 'text': insights, 'plot': plot_data})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/logistic_regression_insights')
def logistic_regression_insights():
    try:
        global df
        global target
        global numeric_cols
        unique_values = df.select(target).distinct().count()
        if unique_values <= 10:
            insights, plot_data = ig.logisticRegression(df, target)
            return jsonify({'result': 'success', 'text': insights, 'plot': plot_data})
        else:
            return jsonify({'result': 'linear'})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/decisionTree_insights')
def decisionTree_insights():
    try:
        global df
        global target
        insight = ig.decisionTreeClassifier(df, target)
        return jsonify({'result': 'success', 'text': insight})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/randomForest_insights')
def randomForest_insights():
    try:
        global df
        global target
        insight = ig.randomForestClassifier(df, target)
        return jsonify({'result': 'success', 'text': insight})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


@app.route('/insight.html/linear_regression_insights')
def linear_regression_insights():
    try:
        global df
        global target
        insight = ig.linearRegression(df, target)
        return jsonify({'result': 'success', 'text': insight})
    except Exception as e:
        return jsonify({'result': 'failure', 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
