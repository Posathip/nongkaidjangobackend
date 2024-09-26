from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
import cv2
import base64
import os
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import json
import boto3
from storages.backends.s3boto3 import S3Boto3Storage
from botocore.exceptions import NoCredentialsError
from django.http import HttpResponse
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image , ImageEnhance, ImageFilter
import requests
from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image



model = None


@csrf_exempt
@require_http_methods(["POST"])
def mainprocess(request):
    try:
        
        data = json.loads(request.body)

        
        reply_token = data.get('replyToken')
        timestamp = data.get('timestamp')
        filename = data.get('filename')
        image_url = data.get('imageurl')
      
        
        if image_url:
            # Fetch the image from the URL
            response = requests.get(image_url)
            if response.status_code == 200:
                # Decode the image data from HTTP to base64

                # image_data = base64.b64encode(response.content).decode('utf-8')
               if load_keras_model():
                    try:
                        print('A')
                        image_tensor = preprocess_image_from_url(image_url)
                        result   = predict_image(image_tensor)
                        save_image_from_url(image_url, filename)
                        # print(f"Image Array Shape: {img_array.shape}")
                        # print(f"Base64 Encoded Image Data: {image_data[:100]}...")
                          
                        response_message = {
                        'totalEgg': 4,
                        'fertile': 10,
                        'infertile': 20,
                        'result': result,
                        'prediction': '18',
                        'replyToken': reply_token,
                        'timestamp': timestamp,
                        'filename': filename,
                        'imageResult':  f'https://agtechai.in.th/eggmobiledjango/static/images/${filename}',
                        'message': "Process Complete"
                        }
                   
                    except Exception as e:
                     return JsonResponse({'error': str(e)}, status=500)
            
                # Prepare the file path
        #         file_path = os.path.join('static', 'images', filename)
        #         os.makedirs(os.path.dirname(file_path), exist_ok=True)
        #         # Save the image to the static folder
        #         with open(file_path, 'wb') as file:
        #             file.write(base64.b64decode(image_data))
                
              
        #         # response_message = {
        #         #         'totalEgg': 4,
        #         #         'fertile': 10,
        #         #         'infertile': 20,
                      
        #         #         'replyToken': reply_token,
        #         #         'timestamp': timestamp,
        #         #         'filename': filename,
        #         #         'imageResult':  image_url,
        #         #         'message': f"Image saved to static folder at {file_path}"
        #         #         }

                
        #             response_message = {
        #              'totalEgg': 4,
        #                 'fertile': 10,
        #                 'infertile': 20,
                      
        #             'replyToken': reply_token,
        #             'timestamp': timestamp,
        #             'filename': filename,
        #             'imageResult':  image_url,
        #                 'message': "Complete"
        #             }

                  
            else:
                response_message = {'error': 'Failed to fetch image from URL'}
        else:
            response_message = {'error': 'No image URL provided'}

    except json.JSONDecodeError:
        # Handle invalid JSON
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    return JsonResponse(response_message)


        

def home(request):
    return HttpResponse("Welcome to the website!")


def load_keras_model():
    global model
    try:
        # Load the model from the static/models directory
        model_path = os.path.join(settings.BASE_DIR, 'static', 'model', 'my_model.h5')
        model = load_model(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading the model: {e}")
        model = None
        return False



def preprocess_image_from_url(image_url):
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            img = img.resize((300, 300))  # Adjust according to your model's input size
            img = img.convert('RGB')
            x = keras_image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            image_tensor = np.vstack([x])
            return image_tensor

         
        except Exception as e:
            raise ValueError(f"Error processing image from URL: {e}")



def predict_image(image_tensor):
    try:
       
        predictions = model.predict(image_tensor)
        prediction = predictions[0][0]

        # Define your threshold here
        threshold = 0.5
        print(predictions)
        if prediction > threshold:
            result = "human"
        else:
            result = "horse"

        return result 
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def save_image_from_url(image_url, filename):
    # Define the file path in the static directory
    file_path = os.path.join(settings.BASE_DIR, 'static', 'images', filename)
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Download the image from the URL
    response = requests.get(image_url)
    
    # Check if the request was successful
   
    if response.status_code == 200:
        # Open the image using Pillow
        image = Image.open(BytesIO(response.content))
        
        # Convert the image to RGB mode if it is not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Modify the image (e.g., rotating, resizing, enhancing)
        
        # 1. Rotate the image by 15 degrees
        image = image.rotate(15)
        
        # 2. Resize the image to 300x300 pixels (keeping aspect ratio can be done with .thumbnail())
        image = image.resize((300, 300))
        
        # 3. Enhance the brightness by 1.5x
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.5)
        
        # 4. Optionally, apply a slight blur to the image
        image = image.filter(ImageFilter.GaussianBlur(1))
        
        # Save the image to the file
        image.save(file_path)
        
    else:
        # Handle the error if the request failed
        raise Exception(f"Failed to download image. Status code: {response.status_code}")

# Call the model loading function when the server starts or the first request happens
if model is None:
    load_keras_model()