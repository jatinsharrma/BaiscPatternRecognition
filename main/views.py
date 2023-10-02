from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
import csv
from copy import copy
import main.helpers as helpers

def uploadFile(request):
    return render(request,'main/index.html')

def featureSelection(request):
    data = {}
    if request.method == "POST":
        file = request.FILES["file"]
        if not file.name.endswith('csv'):
            data['error'] = "Unsupported file format"
        else:
            df = helpers.convertToPandas(file)
            data = helpers.extractFeaturesNames(df)
            class_name = helpers.extractClassName(df)
            if request.session.has_key("df"):
                del request.session["df"]
            request.session["df"] = df.to_csv()
            request.session["class_name"] = class_name

    return render(request,'main/feature_selection.html',context=data)

def showDataset(request):
    data = {}
    if request.method=="POST":
        # extrating selected features from POST request
        data["features"] = {"names":[],"Numerical":[],"Quantitative":[]}
        features = []
        request_data = dict(request.POST)
        if "numerical" in request_data:
            features = request_data["numerical"]
            data["features"]["Numerical"] = copy(features)
        if "quantitative" in request_data:
            features.extend(request_data["quantitative"])
            data["features"]["Quantitative"] = request_data["quantitative"]
        data["features"]["name"] = features
        # extracting required data from the uploaded CSV which is saved on session memory
        if request.session.has_key("df"):
            columns = [request.session["class_name"]]
            columns.extend(features)
            df = helpers.convertToPandas(request.session["df"],columns)
            request.session["df"] = df.to_csv()
            request.session["features"] = data["features"]
            data["df_10"] = helpers.get10RowsOfData(df)
    else:
        if request.session.has_key("df"):
            data["features"] = request.session["features"]
            df = helpers.convertToPandas(request.session["df"],data["features"]["name"])
            data["df_10"] = helpers.get10RowsOfData(df)
    return render(request,'main/datasetview.html',context=data)

def classVizRepresentation(request):
    data = {}
    if request.session.has_key("df"):
        if request.session.has_key("features"):
            if request.session.has_key("class_name"):
                columns = [request.session["class_name"]]
                columns.extend(request.session["features"]["Numerical"])
                print(columns)
                df = helpers.convertToPandas(request.session["df"],columns)
                data["class"] = helpers.classVizData(df,columns)
                return render(request,'main/classView.html',data)
    return render(request,'main/error.html',context=data)

def featuresVizRepresentation(request):
    data = {}
    if request.session.has_key("df"):
        if request.session.has_key("features"):
            if request.session.has_key("class_name"):
                columns = [request.session["class_name"]]
                columns.extend(request.session["features"]["Numerical"])
                df = helpers.convertToPandas(request.session["df"],columns)
                data["feature"] = helpers.featureVizDataNew(df,columns)
                return render(request,'main/featuresView.html',data)
    return render(request,'main/error.html',context=data)

def boxPlotRepesentation(request):
    data = {}
    if request.session.has_key("df"):
        if request.session.has_key("features") and len(request.session["features"]["Numerical"]) > 0:
            if request.session.has_key("class_name"):
                columns = [request.session["class_name"]]
                columns.extend(request.session["features"]["Numerical"])
                df = helpers.convertToPandas(request.session["df"],columns)
                data["boxplot"] = helpers.plotBoxPlot(df, columns)
                return render(request,'main/boxPlot.html',context=data)
    return render(request,'main/error.html',context=data)

def predict(request):
    data = {}
    if request.method == "POST":
        if request.session.has_key("df"):
            point = request.POST["point"]
            columns = [request.session["class_name"]]
            columns.extend(request.session["features"]["Numerical"])
            df = helpers.convertToPandas(request.session["df"],columns)
            numerical = helpers.predict(df,point[:len(columns)])
            data["confusion"] = helpers.confusionMatix(df)
            
            columns = [request.session["class_name"]]
            columns.extend(request.session["features"]["Quantitative"])
            if len(columns) > 1:
                df = helpers.convertToPandas(request.session["df"],columns)
                quantitative = helpers.bayesAnalysis(df,point.split(",")[-len(columns)+1:])
            else:
                quantitative = {item:1 for item in numerical}

            data["predict"] = helpers.transforDataForDisplay(numerical,quantitative)

            return render(request,'main/predict.html',context=data)
    if request.session.has_key('features'):
        data["placeholder"] = " ".join(request.session["features"]['name'])
    return render(request,'main/predict.html',context=data)