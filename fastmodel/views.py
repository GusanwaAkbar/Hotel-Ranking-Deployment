from django.shortcuts import render


# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import FastmodelConfig


#Okay, let's go ahead and start using these new components to refactor our views slightly.


    

class call(APIView):

    def get(self,request):
        if True:
            
            # sentence is the query we want to get the prediction for
            #params =  request.GET.get('sentence')
            
            # predict method used to get the prediction
            response = FastmodelConfig.final_data
            # returning JSON response
            return JsonResponse(response,safe=False)


class call2(APIView):

    def get(self,request):
        if True:
            
            # sentence is the query we want to get the prediction for
            params =  request.GET.get()

            if request.method == 'POST':
                serializer = SnippetSerializer(data=request.data)
                if serializer.is_valid():
                    serializer.save()
                    return Response(serializer.data, status=status.HTTP_201_CREATED)
                
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            


            