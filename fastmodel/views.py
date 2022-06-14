from django.shortcuts import render


# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import FastmodelConfig

class call(APIView):

    def get(self,request):
        if True:
            
            # sentence is the query we want to get the prediction for
            #params =  request.GET.get('sentence')
            
            # predict method used to get the prediction
            response = FastmodelConfig.final_data
            # returning JSON response
            return JsonResponse(response,safe=False)