from django.http import HttpResponse
from rest_framework.response import Response
from rest_framework.decorators import api_view
from PIL import Image
import os
import io
import base64
from django.shortcuts import render
from django.utils.safestring import mark_safe

@api_view(['GET'])
def getData(request):

    # with open("kat.jpg", "rb") as imagefile:
    #     convert = base64.b64encode(imagefile.read())
    
    img = Image.open('kat.jpg')
    response = HttpResponse(content_type='image/jpg')
    img.save(response, "JPEG")
      
          
    return response

