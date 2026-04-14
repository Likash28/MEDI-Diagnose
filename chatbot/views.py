import os

# Suppress TensorFlow logs and oneDNN overhead before any TF import
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import json
import re
import io
import base64
import pickle

import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image

# Django
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
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib.auth.decorators import login_required

# ── Lazy model cache ──────────────────────────────────────────────────────────
# Heavy libraries (tensorflow, transformers, torch) are imported inside these
# loader functions so they do NOT run at server startup — only on first use.

_tokenizer = None
_bert_model = None
_segmentation_model = None
_pneumonia_model = None
_diabetes_model = None


def get_bert():
    global _tokenizer, _bert_model
    if _tokenizer is None:
        from transformers import BertTokenizer, BertModel
        _tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        _bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
    return _tokenizer, _bert_model


def get_segmentation_model():
    global _segmentation_model
    if _segmentation_model is None:
        import tensorflow as tf
        _segmentation_model = tf.keras.models.load_model(
            'deeplabnetown.h5',
            custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
        )
    return _segmentation_model


def get_pneumonia_model():
    global _pneumonia_model
    if _pneumonia_model is None:
        import tensorflow as tf
        _pneumonia_model = tf.keras.models.load_model('saved_models.h5')
    return _pneumonia_model


def get_diabetes_model():
    global _diabetes_model
    if _diabetes_model is None:
        _diabetes_model = pickle.load(open('diabetes-model.pkl', 'rb'))
    return _diabetes_model


# ── Auth views ────────────────────────────────────────────────────────────────

@csrf_protect
def home(request):
    if request.session.get('email'):
        return render(request, 'index.html')
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

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists! Please try another.")
            return redirect('home')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('home')

        if len(username) > 20:
            messages.error(request, "Username must be under 20 characters.")
            return redirect('home')

        if pass1 != pass2:
            messages.error(request, "Passwords didn't match.")
            return redirect('home')

        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name = fname
        myuser.last_name = lname
        myuser.is_active = False
        myuser.save()

        messages.success(request, "Account created. Check your email to activate it.")

        subject = "Welcome to Medi Diagnos!"
        message = (f"Hello {myuser.username}!\n\nWelcome to Medi Diagnos!\n"
                   "Thank you for registering. Please confirm your email to activate your account.\n\n"
                   "Thanks,\nMedi Diagnos")
        send_mail(subject, message, settings.EMAIL_HOST_USER, [myuser.email], fail_silently=True)

        current_site = get_current_site(request)
        email_subject = "Confirm your Medi Diagnos email"
        message2 = render_to_string('email_confirmation.html', {
            'name': myuser.first_name,
            'domain': current_site.domain,
            'uid': urlsafe_base64_encode(force_bytes(myuser.pk)),
            'token': generate_token.make_token(myuser)
        })
        email_msg = EmailMessage(email_subject, message2, settings.EMAIL_HOST_USER, [myuser.email])
        email_msg.fail_silently = True
        email_msg.send()

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
        myuser.save()
        login(request, myuser)
        messages.success(request, "Account activated!")
        return redirect('login')

    return render(request, 'activation_failed.html')


@csrf_protect
def user_login(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        user = authenticate(username=username, password=pass1)
        if user is not None:
            login(request, user)
            request.session['myuser'] = username
            request.session['email'] = user.email
            request.session['username'] = user.username
            return redirect('index')
        messages.error(request, "Invalid credentials.")
        return redirect('home')

    return render(request, "login.html")


@csrf_protect
def user_logout(request):
    request.session.pop('email', None)
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('home')


@csrf_protect
@login_required
def index(request):
    if 'email' in request.session:
        return render(request, 'index.html')
    return redirect("login")


@csrf_protect
@login_required
def appoint(request):
    if request.method == "POST":
        gmail = request.POST['gmailid']
        subject = request.POST['subject']
        msg_body = request.POST['messages']
        username = request.user.username

        send_mail(
            subject,
            f"From: {username}\nEmail: {gmail}\nMessage: {msg_body}",
            settings.EMAIL_HOST_USER,
            ['likashgunisetti@gmail.com'],
            fail_silently=True
        )
        send_mail(
            "Welcome to Medi-Diagnos",
            f"Hi {username},\n\nThanks for your message! We'll get back to you soon.\n\nMedi-Diagnos",
            settings.EMAIL_HOST_USER,
            [gmail],
            fail_silently=True
        )
        return render(request, 'sent.html')

    return render(request, 'index.html')


# ── Chatbot ───────────────────────────────────────────────────────────────────

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        from .chat import get_response
        data = json.loads(request.body.decode('utf-8'))
        text = data['message']
        response = get_response(text)
        return JsonResponse({"answer": response})
    return render(request, 'chatbot.html')


# ── Protein Stability ─────────────────────────────────────────────────────────

complementary_nucleotides = {"G": "C", "C": "G", "A": "U", "T": "A"}


def protein_to_dna(sequence):
    codon_table = {
        'I': 'ATA', 'M': 'ATG', 'T': 'ACA', 'N': 'AAC', 'K': 'AAA',
        'S': 'AGC', 'R': 'AGA', 'L': 'CTA', 'P': 'CCA', 'H': 'CAC',
        'Q': 'CAA', 'V': 'GTA', 'A': 'GCA', 'D': 'GAC', 'E': 'GAA',
        'G': 'GGA', 'C': 'TGC', 'W': 'TGG', 'F': 'TTC', 'Y': 'TAC',
    }
    return ''.join(codon_table.get(aa, 'NNN') for aa in sequence)


def convert_to_dense_columns(features_array):
    df = pd.DataFrame(features_array)
    df.columns = ['Feature_' + str(x) for x in df.columns]
    return df


def return_amino_acid_df(df):
    search_amino = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    for aa in search_amino:
        df[aa] = df['protein_sequence'].str.count(aa, re.I)
    return df


def predict_stability(request):
    input_sequences, ph_values, Temp_values, DNA_values, mRNA_values = [], [], [], [], []

    if request.method == 'POST':
        try:
            tokenizer, bert_model = get_bert()
            l = request.POST['ps']
            p = request.POST['ph']
            df1 = pd.DataFrame({'protein_sequence': [l], 'ph': [p]})

            sequence_example = re.sub(r"[UZOB]", "X", l)
            encoded_input = tokenizer(
                sequence_example, add_special_tokens=True,
                padding=True, is_split_into_words=True, return_tensors="pt"
            )
            import torch
            with torch.no_grad():
                output = bert_model(**encoded_input)
                embedding = output[1].detach().cpu().numpy()[0]

            train_feats_df = convert_to_dense_columns([embedding])
            train_feats_df["protein_length"] = len(l)
            df1 = return_amino_acid_df(df1)
            df1 = pd.get_dummies(df1, columns=['ph'])
            df1.drop(columns=["protein_sequence"], inplace=True)
            maindf = pd.concat([df1, train_feats_df], axis=1)

            pickled_model = joblib.load('lik.pkl')
            Temp = pickled_model.predict(maindf)[0]
            DNA = protein_to_dna(l)
            DNA = DNA.translate(str.maketrans({'G': 'C', 'C': 'G', 'A': 'U', 'T': 'A'}))
            mRNA = "".join(complementary_nucleotides.get(nt, "") for nt in DNA)

            input_sequences.append(l)
            ph_values.append(p)
            Temp_values.append(Temp)
            DNA_values.append(DNA)
            mRNA_values.append(mRNA)
        except Exception as e:
            messages.error(request, f"Prediction error: {str(e)}")

    return render(request, 'mrn.html', {
        'input_sequences': input_sequences,
        'ph_values': ph_values,
        'Temp_values': Temp_values,
        'DNA_values': DNA_values,
        'mRNA_values': mRNA_values,
    })


# ── Pneumonia Detection ───────────────────────────────────────────────────────

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    return img.reshape(-1, 150, 150, 1)


def predict_pneumonia(request):
    results = []

    if request.method == 'POST' and 'user_images' in request.FILES:
        try:
            model = get_pneumonia_model()
            user_images = request.FILES.getlist('user_images')
            input_folder = os.path.join(settings.MEDIA_ROOT, 'input_images')
            os.makedirs(input_folder, exist_ok=True)

            for user_image in user_images:
                input_filename = f"input_{user_image.name}"
                input_image_path = os.path.join(input_folder, input_filename)

                with open(input_image_path, 'wb') as f:
                    for chunk in user_image.chunks():
                        f.write(chunk)

                processed = preprocess_image(input_image_path)
                prediction = model.predict(processed)
                result = "PNEUMONIA NEGATIVE" if prediction > 0.5 else "PNEUMONIA POSITIVE"
                results.append((
                    os.path.join(settings.MEDIA_URL, 'input_images', input_filename),
                    result
                ))
        except Exception as e:
            messages.error(request, f"Model error: {str(e)}")

    return render(request, 'pneumonia.html', {'results': results})


# ── Brain MRI Segmentation ────────────────────────────────────────────────────

smooth = 1e-15

def dice_coef(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def segment_image(image_np):
    H, W = 256, 256
    image = cv2.resize(image_np, (W, H))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)
    model = get_segmentation_model()
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= 0.5).astype(np.uint8) * 255
    return y_pred


def brain_mri_segmentation(request):
    input_images, mask_images = [], []

    if request.method == 'POST':
        try:
            for image in request.FILES.getlist('images'):
                image_data = image.read()
                image_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                mask = segment_image(image_np)

                input_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
                mask_pil = Image.fromarray(mask)

                in_buf, out_buf = io.BytesIO(), io.BytesIO()
                input_pil.save(in_buf, format='JPEG')
                mask_pil.save(out_buf, format='JPEG')

                input_images.append(base64.b64encode(in_buf.getvalue()).decode('utf-8'))
                mask_images.append(base64.b64encode(out_buf.getvalue()).decode('utf-8'))
        except Exception as e:
            messages.error(request, f"Segmentation error: {str(e)}")

    return render(request, 'brainmra.html', {'image_pairs': zip(input_images, mask_images)})


# ── Diabetes Prediction ───────────────────────────────────────────────────────

def predict_diabetes(request):
    if request.method == 'POST':
        try:
            model = get_diabetes_model()
            preg     = float(request.POST.get('pregnancies'))
            glucose  = float(request.POST.get('glucose'))
            bp       = float(request.POST.get('bloodpressure'))
            st       = float(request.POST.get('skinthickness'))
            insulin  = float(request.POST.get('insulin'))
            bmi      = float(request.POST.get('bmi'))
            dpf      = float(request.POST.get('dpf'))
            age      = float(request.POST.get('age'))

            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            my_prediction = int(model.predict(data)[0])
            s = "danger" if my_prediction == 1 else "safe"

            return render(request, 'diabetes.html', {
                'preg': preg, 'a1': glucose, 'a2': bp, 'a3': st,
                'a4': insulin, 'a5': bmi, 'a6': dpf, 'a7': age,
                'prediction': my_prediction, 'prediction_text': s
            })
        except FileNotFoundError:
            messages.error(request, "Diabetes model file not found.")
        except Exception as e:
            messages.error(request, f"Prediction error: {str(e)}")

    return render(request, 'diabetes.html')
