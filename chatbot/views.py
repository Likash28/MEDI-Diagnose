import numpy as np
import torch
import json
import joblib
import os
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.core.mail import EmailMessage, send_mail
from chatbot import settings
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth import authenticate, login, logout
from .tokens import generate_token
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_protect
from django.core.files.storage import FileSystemStorage
from PIL import ImageFont, Image
import tensorflow as tf

# from object_detection.utils import ops as utils_ops
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util
from transformers import BertModel, BertTokenizer
import re
import pandas as pd
from xgboost import XGBRegressor
import joblib
import cv2
from keras.models import load_model
# from object_detection.utils import label_map_util
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import io
import base64
from .chat import get_response




from django.shortcuts import render

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        text = data['message']
        response = get_response(text)
        print(response)
        message = {"answer": response}
        return JsonResponse(message)
    else:
        return render(request, 'chatbot.html')



    
    
@csrf_protect
def home(request):
    if request.session.get('email'):
        return render(request, 'index.html')
    else:
        return redirect('register')
@csrf_protect
def register(request):
    if request.method == "POST":
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
        
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username.")
            return redirect('home')
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email Already Registered!!")
            return redirect('home')
        
        if len(username)>20:
            messages.error(request, "Username must be under 20 charcters!!")
            return redirect('home')
        
        if pass1 != pass2:
            messages.error(request, "Passwords didn't matched!!")
            return redirect('home')
        
        # if not username.isalnum():
        #     messages.error(request, "Username must be Alpha-Numeric!!")
        #     return redirect('home')
        

        myuser = User.objects.create_user(username, email, pass1)
        myuser.email = email
        myuser.first_name = fname
        myuser.last_name = lname
        # myuser.is_active = False
        myuser.is_active = False
        myuser.save()
        messages.success(request, "Your Account has been created succesfully!! Please check your email to confirm your email address in order to activate your account.")
        
        # Welcome Email
        subject = "Welcome to Medi Diagnos-Login!!"
        message = "Hello " + myuser.username + "!! \n" + "Welcome to Medi Diagnos!! \nThank you for visiting our website\n We have also sent you a confirmation email, please confirm your email address. \n\nThanking You\nMedi Diagnos"        
        from_email = settings.EMAIL_HOST_USER
        to_list = [myuser.email]
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
        # Email Address Confirmation Email
        current_site = get_current_site(request)
        email_subject = "Confirm your Medi Diagnos-Login!!"
        message2 = render_to_string('email_confirmation.html',{
            
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser)
        })
        email = EmailMessage(
        email_subject,
        message2,
        settings.EMAIL_HOST_USER,
        [myuser.email],
        )
        email.fail_silently = True
        email.send()
        
        return redirect('login')
        
        
    return render(request, "register.html")

@csrf_protect
def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        myuser = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        myuser = None

    if myuser is not None and generate_token.check_token(myuser, token):
        myuser.is_active = True
        # user.profile.signup_confirmation = True
        myuser.save()
        login(request, myuser)
        messages.success(request, "Your Account has been activated!!")
        return redirect('login')
    else:
        return render(request, 'activation_failed.html')

@csrf_protect
def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        check_user = User.objects.filter(username=username, password=pass1)
        if check_user:
            request.session['user'] = username
            return redirect('home')
        user = authenticate(username=username, password=pass1)
        if user is not None:
            login(request, user)
            fname = user.first_name
            # messages.success(request, "Logged In Sucessfully!!")
            # return render(request, "index.html",{"fname":fname})
            request.session['myuser'] = username
            request.session['email'] =user.email
            request.session['username']=user.username
            return redirect('index')
        else:
            messages.error(request, "Bad Credentials!!")
            return redirect('home')
    
    return render(request, "login.html")

@csrf_protect
def user_logout(request):
    del request.session['email']
    logout(request)
    messages.success(request, "Logged Out Successfully!!")
    return redirect('home')



@csrf_protect
@login_required
def appoint(request):
    if request.method == "POST":
        gmail = request.POST['gmailid']
        subject = request.POST['subject']
        messages = request.POST['messages']
        user_email = request.user.email
        username = request.user.username
        # Welcome Email
        subject =  subject
        message = "from " +username+" \n" +"Email: " +gmail+" \n"  +"Message : "+ messages      
        from_email = settings.EMAIL_HOST_USER
        to_list =  ['likashgunisetti@gmail.com']
        send_mail(subject, message, from_email, to_list, fail_silently=True)
        
        # Welcome Email
       # Welcome Email
        subject = "Welcome to Medi-Diagnos"
        message = "Hi " + username + "\n" + "Thanks For Your Feedback and we will get back to you soon!!\nThanks for spending your valuable time with our website."
        from_email = settings.EMAIL_HOST_USER
        to_list = [gmail]  # Use the email provided in the form
        send_mail(subject, message, from_email, to_list, fail_silently=True)

        return render(request, 'sent.html')
    

@csrf_protect
@login_required
def index(request):
    if 'email' in request.session:
        return render(request, 'index.html')
    else:
        return redirect("login")

#Preprocessing
    

# Patch tf1 into `utils.ops`
# utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model():
    model_dir = "saved_model"
    model = tf.compat.v2.saved_model.load(model_dir, None)
    model = model.signatures['serving_default']
    return model

# Load the object detection model
detection_model = load_model()

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'object-detection.pbtxt'
# category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# def run_inference_for_single_image(model, image):
#     image = np.asarray(image)
#     input_tensor = tf.convert_to_tensor(image)
#     input_tensor = input_tensor[tf.newaxis, ...]
#     output_dict = model(input_tensor)

#     num_detections = int(output_dict.pop('num_detections'))
#     output_dict = {key: value[0, :num_detections].numpy()
#                    for key, value in output_dict.items()}
#     output_dict['num_detections'] = num_detections

#     # detection_classes should be ints.
#     output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

#     if 'detection_masks' in output_dict:
#         # detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
#             output_dict['detection_masks'], output_dict['detection_boxes'],
#             image.shape[0], image.shape[1]
#             detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
#             output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

#     return output_dict

# def process_image(image_path):
#     image_np = np.array(Image.open(image_path))
#     output_dict = run_inference_for_single_image(detection_model, image_np)
#     vis_util.visualize_boxes_and_labels_on_image_array(
#         image_np,
#         output_dict['detection_boxes'],
#         output_dict['detection_classes'],
#         output_dict['detection_scores'],
#         category_index,
#         instance_masks=output_dict.get('detection_masks_reframed', None),
#         use_normalized_coordinates=True,
#         line_thickness=8)
#     return Image.fromarray(image_np)



tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model111 = BertModel.from_pretrained("Rostlab/prot_bert")

# Define a dictionary for complementary nucleotides
complementary_nucleotides = {"G": "C", "C": "G", "A": "U", "T": "A"}


def protein_to_dna(l):
    codon_table = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'', 'TAG':'',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
    }
    dna_sequence = ""
    for amino_acid in l:
        for codon, aa in codon_table.items():
            if aa == amino_acid:
                dna_sequence += codon
    return dna_sequence

def convert_to_dense_columns(features_array):
    df = pd.DataFrame(features_array)
    df.columns = ['Feature_' + str(x) for x in df.columns]
    return df

def return_amino_acid_df(df):
    search_amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for amino_acid in search_amino:
        df[amino_acid] = df['protein_sequence'].str.count(amino_acid, re.I)
    return df





# Models

# def dental_view(request):
#     input_images = []
#     output_images = []

#     if request.method == 'POST' and 'files[]' in request.FILES:
#         uploaded_files = request.FILES.getlist('files[]')

#         # Define the input_images_folder here
#         input_images_folder = os.path.join(settings.MEDIA_ROOT, 'input_images')
#         os.makedirs(input_images_folder, exist_ok=True)

#         for uploaded_file in uploaded_files:
#             # Your processing logic for each file goes here

#             # Generate unique filenames for the input and processed images
#             input_filename = f"input_{uploaded_file.name}"
#             processed_filename = f"processed_{uploaded_file.name}"

#             # Create the full paths for the input and processed images
#             input_image_path = os.path.join(input_images_folder, input_filename)
#             processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_images', processed_filename)

#             # Save the uploaded image to the input images folder
#             with open(input_image_path, 'wb') as destination:
#                 for chunk in uploaded_file.chunks():
#                     destination.write(chunk)

#             # Process the uploaded image
#             output_image = process_image(input_image_path)

#             # Save the processed image with the same filename
#             output_image.save(processed_image_path)

#             # Add the input and processed image paths to their respective lists
#             input_images.append(os.path.join(settings.MEDIA_URL, 'input_images', input_filename))
#             output_images.append(os.path.join(settings.MEDIA_URL, 'processed_images', processed_filename))

#     return render(request, 'dental.html', {'input_images': input_images, 'output_images': output_images})

def predict_stability(request):
    input_sequences = []
    ph_values = []
    Temp_values = []
    DNA_values = []
    mRNA_values = []

    if request.method == 'POST':
        l = request.POST['ps']
        p = request.POST['ph']
        data = {'protein_sequence': l, 'ph': p}
        df1 = pd.DataFrame(data, index=[0])
    
        # Process the uploaded data
        embeddings_list = []
        sequence_Example = l
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = tokenizer(sequence_Example, add_special_tokens=True, padding=True,
                                  is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            output = model111(**encoded_input)
            output = output[1].detach().cpu().numpy()[0]
            embeddings_list.append(output)
        train_features = embeddings_list
        train_feats_df = convert_to_dense_columns(train_features)
        train_feats_df["protein_length"] = len(l)
        df1 = return_amino_acid_df(df1)
        
        # One-hot encode the 'ph' column
        df1 = pd.get_dummies(df1, columns=['ph'])
        # Drop the 'protein_sequence' column
        df1.drop(columns=["protein_sequence"], inplace=True)
        maindf = pd.concat([df1, train_feats_df], axis=1)
        pickled_model = joblib.load('lik.pkl')
        Temp = pickled_model.predict(maindf)[0]
        DNA = protein_to_dna(l)
        DNA = DNA.translate(str.maketrans({'G': 'C', 'C': 'G', 'A': 'U', 'T': 'A'}))
        mRNA = "".join(complementary_nucleotides.get(nt, "") for nt in DNA)

        # Store the results in lists
        input_sequences.append(l)
        ph_values.append(p)
        Temp_values.append(Temp)
        DNA_values.append(DNA)
        mRNA_values.append(mRNA)

    return render(request, 'mrn.html', {
        'input_sequences': input_sequences,
        'ph_values': ph_values,
        'Temp_values': Temp_values,
        'DNA_values': DNA_values,
        'mRNA_values': mRNA_values,
    })


# Define the preprocess_image function
def preprocess_image(image):
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = img.reshape(-1, 150, 150, 1)
    return img




def predict_pneumonia(request):
    results = []

    if request.method == 'POST' and 'user_images' in request.FILES:
        user_images = request.FILES.getlist('user_images')

        for user_image in user_images:
            # Create a path to the folder where input images will be stored
            input_images_folder = os.path.join(settings.MEDIA_ROOT, 'input_images')
            os.makedirs(input_images_folder, exist_ok=True)

            # Generate a unique filename for the input image
            input_filename = f"input_{user_image.name}"
            input_image_path = os.path.join(input_images_folder, input_filename)

            # Save the uploaded image to the input images folder
            with open(input_image_path, 'wb') as destination:
                for chunk in user_image.chunks():
                    destination.write(chunk)

            # Preprocess the image
            processed_image = preprocess_image(input_image_path)

            # Load your trained model using tf.keras.models.load_model
            model = tf.keras.models.load_model('saved_models.h5')

            # Make a prediction
            prediction = model.predict(processed_image)

            # Determine the class based on the prediction
            if prediction > 0.5:
                result = "PNEUMONIA NEGATIVE"
            else:
                result = "PNEUMONIA POSITIVE"

            # Append the image path and result to the results list as a tuple
            results.append((os.path.join(settings.MEDIA_URL, 'input_images', input_filename), result))

    return render(request, 'pneumonia.html', {'results': results})


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
segmentation_model=tf.keras.models.load_model('deeplabnetown.h5', custom_objects={'dice_loss':                   
dice_loss,'dice_coef':dice_coef})
H = 256
W = 256
save_image_path='mini code'
def read_image(image):
     ## [H, w, 3]
    image = cv2.resize(image, (W, H))       ## [H, w, 3]
    x = image/255.0                         ## [H, w, 3]
    x = np.expand_dims(x, axis=0)
    y_pred = segmentation_model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred * 255
    return y_pred

def brain_mri_segmentation(request):
    input_images = []
    mask_images = []

    if request.method == 'POST':
        images = request.FILES.getlist('images')  # Get a list of uploaded images

        for image in images:
            # Process each uploaded image and get the segmentation mask
            # Replace this code with your actual processing logic
            image_data = image.read()
            image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            segmentation_mask = read_image(image_np)

            # Convert images to base64 for rendering in HTML
            input_pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            output_pil_image = Image.fromarray(segmentation_mask)

            input_buffer = io.BytesIO()
            output_buffer = io.BytesIO()
            input_pil_image.save(input_buffer, format='JPEG')
            output_pil_image.save(output_buffer, format='JPEG')

            input_images.append(base64.b64encode(input_buffer.getvalue()).decode('utf-8'))
            mask_images.append(base64.b64encode(output_buffer.getvalue()).decode('utf-8'))

    # Zip the two lists together for easy iteration in the template
    image_pairs = zip(input_images, mask_images)

    return render(request, 'brainmra.html', {'image_pairs': image_pairs})

import pickle

# loaded_model = pickle.load(open('diabetes-model.pkl', 'rb'))
loaded_model=None


def predict_diabetes(request):
    preg=None
    glucose=None
    bp=None
    st=None
    insulin=None
    bmi=None
    dpf=None
    age=None
    my_prediction=None
    s=None
    if request.method == 'POST':
        preg = float(request.POST.get('pregnancies'))
        glucose = float(request.POST.get('glucose'))
        bp = float(request.POST.get('bloodpressure'))
        st = float(request.POST.get('skinthickness'))
        insulin = float(request.POST.get('insulin'))
        bmi = float(request.POST.get('bmi'))
        dpf = float(request.POST.get('dpf'))
        age = float(request.POST.get('age'))
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = float(loaded_model.predict(data))
        
        s = "danger" if my_prediction == 1 else "safe"

        return render(request, 'diabetes.html', {'preg': preg,'a1': glucose,
            'a2': bp,
            'a3': st,
            'a4': insulin,
            'a5': bmi,
            'a6': dpf,
            'a7': age,
            'prediction': my_prediction,
            'prediction_text': s
        })
    else:
        return render(request, 'diabetes.html')
  
