from django.shortcuts import render


# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import FastmodelConfig
import pandas as pd
import numpy as np


#Okay, let's go ahead and start using these new components to refactor our views slightly.


    

class call(APIView):

    def get(self,request):
        if True:
            
            # sentence is the query we want to get the prediction for
            #params =  request.GET.get('sentence')
            
            # predict method used to get the prediction
            response = FastmodelConfig.final_data
            # returning JSON response
            return HttpResponse(response,content_type="text/json-comment-filtered")


class call_2(APIView):

    def post(self,request):
        if True:
            
            # sentence is the query we want to get the prediction fo

            if request.method == 'POST':
                print(request.data['interest'])
                print("good job 2")

                a = request.data['interest']

                facilities_list = []
                for x in a:
                    facilities_list.append(x)

                total_score = FastmodelConfig.count_references_score(facilities_list)

                score = FastmodelConfig.score
                data = FastmodelConfig.data


                data_2 = data
                data_2['Score'] = score + total_score

                sorted_data = data_2.sort_values(by=['Score'], ascending=False)

                final_data = sorted_data.to_json(orient="table")

        
        response = final_data
        
                
        return HttpResponse(response,content_type="text/json-comment-filtered")
            


            