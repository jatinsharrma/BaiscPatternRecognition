from django.urls import path
from . import views

urlpatterns = [
    path('', views.uploadFile, name="main-test"),
    path('showDataSet/', views.showDataset, name="show-dataset"),
    path('featureSelection/',views.featureSelection, name="feature-selection"),
    path('featureView/',views.featuresVizRepresentation, name="feature-view"),
    path('classView', views.classVizRepresentation,name="class-view"),
    path('boxPlot/',views.boxPlotRepesentation, name="box-plot"),
    path('prediction/',views.predict,name="predict"),
]