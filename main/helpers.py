import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal 
from io import StringIO
import plotly.express as px
from sklearn import metrics

def featureVizData(class_viz_data):
    """_summary_
    This function takes the class wise data as a dictionary.
    class_viz_data have class_name as key and as value it has one dictionry named "stats".
    stats have features as key and per feature stats as value.

    This function makes a feature wise data from class wise data by rearranging them.

    Args:
        class_viz_data (dictionary):

    Returns:
        _type_: dictionary
    """
    feature_viz_data = {}
    for class_name in class_viz_data:
        if "stats" in class_viz_data[class_name]:
            for feature,stat in class_viz_data[class_name]["stats"].items():
                if feature not in feature_viz_data:
                    feature_viz_data[feature] ={class_name:{}}
                if class_name not in feature_viz_data[feature]:
                    feature_viz_data[feature][class_name] = {}
                for key, value in stat.items():
                    feature_viz_data[feature][class_name][key] = value
    return feature_viz_data

def extendNestedDictionary(main_dict,small_dict,property_name):
    """_summary_
    
    Args:
        main_dict (dictionary): This is the main dictionary which will be extended
        small_dict (dictionary): This dictionary will be merged in main_dict
        property_name (string): This tell to which property these dictionary belongs to
    """
    for key,value in small_dict.items():
        main_dict[key][property_name] = value

def predict(df, x):
    """_summary_

    Args:
        df (pandas.Dataframe): dataset
        x (list): This a point for which we are finding probability. 

    Returns:
        dictionary : Class wise probabilty
    """
    x = pd.read_csv(StringIO(x),header=None)
    list_of_classes = df.iloc[:, 0].unique()
    results = {}
    for c_name in list_of_classes:
        class_data = df[df.iloc[:, 0] == c_name]
        class_numeric_data = class_data.select_dtypes(include=[np.number])

        class_cov = class_numeric_data.cov(ddof=0)
        class_mean = class_numeric_data.mean()

        y = multivariate_normal.pdf(x, class_mean, class_cov)
        results[c_name] = y
    
    return results

def plotBoxPlot(df,columns):
    """_summary_

    This generates a box plot for the dataframe for required columns

    Args:
        df (pandas.Dataframe): Dataset
        columns (list): List of columns on which box plot will be ploted.

    Returns:
        dictionary: per column html file of boxplot
    """
    result = []
    for i in range(1,len(columns)):
        fig = px.box(
            x = df[columns[0]],
            y = df[columns[i]],
            title="Box plot for : "+ columns[i],
            height=800,
            width=600
        )
        html = fig.to_html()
        result.append(html)
    return result


def extractClassName(df):
    """_summary_

    This function returns class name from the dataframe.
    Considering first column as class name.

    Args:
        df (pandas.Dataframe): dataset

    Returns:
        string : first column header name.
    """
    return df.columns[0]

def extractFeaturesNames(df):
    """_summary_

    This function extract feature names as per the below categories
    1. Numerical
    2. Quantitative
    3. Numerical + Quantitative

    Args:
        df (pandas.Dataframe): dataset

    Returns:
        dictionary : 
            name key : List of all features
            Numerical key : List of all numerical features
            Quantitative key : List of all quantivative features 
    """
    result = {
        "features":{
            "names": list(df.columns[1:]),
            "Numerical" : df.select_dtypes(include=np.number).columns.to_list(),
            "Quantitative" : df.select_dtypes(exclude=np.number).columns.to_list()
        }
    }
    return result

def convertToPandas(raw_data,columns=None):
    """_summary_
    This method converts raw data to pandas DataFrame .

    Args:
        raw_data (Buffer) : data not in pandas dataframe format.
        columns (list, optional): This is used to get specific column as output. Defaults to None.

    Returns:
        pandas.Dataframe : Pandas data frame.
    """
    if columns:
        df = pd.read_csv(StringIO(raw_data))
        return df[columns]
    return pd.read_csv(raw_data)

def get10RowsOfData(df):
    """_summary_
    This function returns 10 rows from given Data Frame

    Args:
        df (pandas.Dataframe): dataset

    Returns:
        lsit : list of rows
    """
    top_10 = df.head(10).to_csv().split("\n")
    new_top_10 = []
    for i in top_10:
        new_top_10.append(i.split(","))
    return new_top_10

def segregateClassVizData(df):
    """_summary_

    This function divides the data frame into classes.

    Args:
        df (pandas.DataFrame): This is the dataset

    Returns:
        dictionary : This will have class name as key and dataframe of just that class as value.
    """
    class_viz_data = {}
    for name,frame in df.groupby(df.columns[0]):
        class_viz_data[name] = frame
    return class_viz_data

def classVizData(df,columns):
    """_summary_

    This function return all stats per class

    Args:
        df (pandas.Dataframe): dataset
        columns (list): List of all columns in dataset

    Returns:
        dictionary: This has class name as key and all stats realted to it as value in the form of dictionary.
    """
    result={}
    class_viz_data = segregateClassVizData(df)
    for c_name in class_viz_data:
        df = class_viz_data[c_name][columns[1:]]
        stats = df.describe().to_dict()

        skew = df.skew()
        extendNestedDictionary(stats,skew,"Skew")

        kurtosis = df.kurt()
        extendNestedDictionary(stats,kurtosis,"Kurtosis")
        result[c_name] = {"stats":stats}

        result[c_name]["covariance"] = {c_name : df.cov().to_dict()}
        result[c_name]["correlation"] = {c_name : df.corr().to_dict()}
    return result

def featureVizDataNew(df,columns):
    """_summary_
    This function convert class wise data dictionary to feature wise dictionary.

    Args:
        df (pandas.Dataframe): dataset
        columns (list): list of all columns in the dataset

    Returns:
        dictionary: feature wise stats 
    """
    result = classVizData(df,columns)
    return featureVizData(result)

def confusionMatix(df):
    """_summary_
    This function generates a confusion matrix.

    Args:
        df (pandas.Dataframe): dataset

    Returns:
        list: list of rows of confusion matrix
    """
    class_name = df.columns[0]
    features = df.select_dtypes(include=np.number).columns.to_list()
    random_df = df.sample(frac=1).sample(frac=0.2)
    actual = []
    predicted = []
    df_dict = random_df.to_dict()
    class_values = []
    for i,j in df_dict[class_name].items():
        x = []
        for f in features:
            x.append(df_dict[f][i])
        if j not in class_values:
            class_values.append(j)
        actual.append(class_values.index(j))
        x = str(x)[1:-1]
        predict_x = predict(df,x)
        winner = [0]*len(features)
        if len(winner)>1:
            for i,j in predict_x.items():
                if not winner:
                    winner[0],winner[1] = i,j
                if winner[1] < j:
                    winner[0],winner[1] = i,j
        if winner[0] not in class_values:
            class_values.append(winner[0])
        predicted.append(class_values.index(winner[0]))
    confusion_matrix = metrics.confusion_matrix(actual, predicted) 
    return confusion_matrix.tolist()

def bayesAnalysis(df,point):
    """_summary_
    This function performs naive Bayes analysis on given data and returns probability that point belongs to each category.

    Args:
        df (pandas.Dataframe) : dataset
        point (list)): list of quantitative point

    Returns:
        dictinary : probability of given point in each class.
    """
    result = {}
    point = point if isinstance(point,tuple) else tuple(point)
    for c_name,df_subset in df.groupby(df.columns[0]):
        temp = df_subset[df_subset.columns[1:]].value_counts(normalize=True).to_dict()
        prob = 0
        if point in temp:
            prob = temp[point]
        else:
            prob = 0
        result[c_name] = prob

    return result

def transforDataForDisplay(numerical,quantitative):
    """_summary_
    This function conbines data from 2 dictinary of same lenght into a list of lists.

    Args:
        numerical (dictionary): probaliites of numerical features.
        quantitative (dictionary): probabilites of quantitative features.

    Returns:
        list: list of list as each sublist having all related probability.
    """
    result = []
    for c_name in numerical:
        temp = [c_name]
        temp.append(numerical[c_name])
        temp.append(0 if c_name not in quantitative else quantitative[c_name])
        temp.append(temp[1]*temp[2])
        for i in range(1,len(temp)):
            temp[i] = float_to_exponential(temp[i])
        result.append(temp)
    return result

def float_to_exponential(num):
    """_summary_
    This function converts the number to exponential form.

    Args:
        num (floating/integer): floating point or integer

    Returns:
        string: exponential representation of given number.
    """
    return f"{num:e}"