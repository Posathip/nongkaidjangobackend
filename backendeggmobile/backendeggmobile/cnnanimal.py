from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
# import cv2
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
def cnnainmal(request):
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
                        label = predict_image(image_tensor)
                        save_image_from_url(image_url, filename)
                        # print(f"Image Array Shape: {img_array.shape}")
                        # print(f"Base64 Encoded Image Data: {image_data[:100]}...")
                          
                        response_message = {
                        'totalEgg': 4,
                        'fertile': 10,
                        'infertile': 20,
                        'result': label,
                        'prediction': '18',
                        'replyToken': reply_token,
                        'timestamp': timestamp,
                        'filename': filename,
                        'imageResult':  f'http://localhost:8000/static/images/{filename}',
                        'message': "Process Complete"
                        }
                    #   # Fetch the image from the URL
                    #     response = requests.get(image_url)
                        
                    #     # Ensure the response content is in bytes
                    #     image_content = response.content
                        
                    #     # Convert image content to PIL Image
                    #     img = Image.open(BytesIO(image_content))
                        
                    #     # Convert image to base64 format
                    #     image_data = base64.b64encode(image_content).decode('utf-8')
                        
                    #     # Resize and normalize the image for prediction
                    #     img = img.resize((224, 224))  # Adjust according to model input size
                    #     img = img.convert('RGB')  # Ensure image is in RGB format
                    #     img_array = np.array(img) / 255.0  # Normalize pixel values
                    #     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
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
        model_path = os.path.join(settings.BASE_DIR, 'static', 'model', 'cnnclassification.h5')
        model = load_model(model_path)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading the model: {e}")
        model = None
        return False



def preprocess_image_from_url(image_url):
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Resize the image to (32, 32) to match the input shape
        img = img.resize((32, 32))

        # Convert the image to RGB (in case it's grayscale or another format)
        img = img.convert('RGB')

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Normalize pixel values to range [0, 1]
        img_array = img_array / 255.0

        # Ensure the shape is (32, 32, 3)
        if img_array.shape != (32, 32, 3):
            raise ValueError(f"Image shape is {img_array.shape}, expected (32, 32, 3)")

        # Add batch dimension to make it (1, 32, 32, 3) for model input
        img_tensor = np.expand_dims(img_array, axis=0)

        return img_tensor
    except Exception as e:
        print(f"Error processing image from URL: {e}")
        return None



def predict_image(image_tensor):
    try:
       
        predictions = model.predict(image_tensor)
        prediction = np.argmax(predictions)
        print(prediction)
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if prediction == 0:
            label = 'airplane'
        elif prediction == 1:
            label = 'automobile'
        elif prediction == 2:
            label = 'bird'
        elif prediction == 3:
            label = 'cat'
        elif prediction == 4:
            label = 'deer'
        elif prediction == 5:
            label = 'dog'
        elif prediction == 6:
            label = 'frog'
        elif prediction == 7:
            label = 'horse'
        elif prediction == 8:
            label = 'ship'
        elif prediction == 9:
            label = 'truck'
        else:
            label = 'Unknown'


        return label
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
        image = image.resize((200, 200))
        
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