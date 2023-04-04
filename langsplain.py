import streamlit as st
from simpletransformers.classification import ClassificationModel
import zipfile
import os
from html2image import Html2Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt
from matplotlib import font_manager

import lime
from lime.lime_text import LimeTextExplainer
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax
#import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer
from transformers import BertTokenizer
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
import re
from torch import argmax
import textwrap
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
plt.rcParams['font.family'] = ['Heiti TC']

def choose_language():
    language = st.selectbox("Relevant Language", ['English', 'Mandarin', 'Malay']) 
    model_type = st.selectbox("Classification Type", ['Emotion', 'Stance']) 
    return language, model_type

def single_input_text():
    col1, col2 = st.columns([1,3])

    with col1:
        named_entity = st.text_input("Please enter a named entity")
    
    with col2:
        relevant_text = st.text_input("Please enter text regarding named entity")
    
    return [relevant_text, named_entity]

def parse_data(data, input_type):

    if input_type == 'Multiple (as a .csv file)':
        df = pd.read_csv(data)
        df.dropna(inplace=True)
        df.name = str(data.name[:-4])
    
    if input_type == 'Single (as a text input)':
        # named_entity, relevant_text = data[1], data[0]
        headers = ['Text', 'Named entity']
        df = pd.DataFrame([data], columns=headers)
        df.name = 'Single_input_' + str(data[1])

    return df


### English ###
def input_english_model_file():
    ml_model_file = st.file_uploader("Upload English ML Model Zip File", type=['zip'])
    return ml_model_file

def input_entities():
    ml_model_file = input_english_model_file()
    data_file = st.file_uploader("Upload English Prediction Data", type=['csv'])
    
    return ml_model_file, data_file
    
def load_ml_model(ml_model_file):

    with zipfile.ZipFile(ml_model_file) as z:
        z.extractall(".")
        # # For checking what files are in the zip folder
        # files = os.listdir(os.curdir + '/' + ml_model_file.name[:-4])
        # for file in files:
        #     st.write(file)

    try:
        import sys
        model_weight_path = "./" + ml_model_file.name[:-4] + '/' + ml_model_file.name[:-4] + '.pth'
        sys.path.insert(0, "./" + ml_model_file.name[:-4])
        from new.utils import get_model
        model = get_model(model_weight_path, 'en')
    except:
        model = ClassificationModel("bert", "./" + ml_model_file.name[:-4], use_cuda = False)
    return model

# def load_ml_model(ml_model_file):
#     with zipfile.ZipFile(ml_model_file) as z:
#          z.extractall(".")
#          #For checking what files are in the zip folder
#          files = os.listdir(os.curdir + '/' + ml_model_file.name[:-4])
#          for file in files:
#               st.write(file)

#     import sys
#     model_weight_path = "./" + ml_model_file.name[:-4] + '/' + ml_model_file.name[:-4] + '.pth'
#     sys.path.insert(0, "./" + ml_model_file.name[:-4])
#     from new.utils import get_model
#     model = get_model(model_weight_path, 'en')
#     return model

def pass_entity(entity, model, model_type):
    if (type(model) == ClassificationModel):
        def predictor(texts):
            texts = list(map(lambda x: [x,entity], texts))
            predictions,raw_outputs = model.predict(texts)
            probas = softmax(raw_outputs, axis =1)
            return probas
    else:
        def predictor(texts):
            texts = list(map(lambda x: [x,entity], texts))
            emotion, stance = model.predict(texts)
            if model_type == 'Stance':
                raw_outputs = stance
            elif model_type == 'Emotion':
                raw_outputs = emotion
            else:
                print('Error: Model has to be of type Stance or Emotion!')
                return
            #This is where model.predict is needed for each input -> returns the actual prediction AND probabilities of each class for individual input
            probas = softmax(raw_outputs, axis =1)
            return probas
    return predictor

def lime_with_model_save_image_all(input, model_type, labels, explainer, model):
    text, entity = input
    storage = []

    for label in labels:
        exp = explainer.explain_instance(text,
                                         classifier_fn = pass_entity(entity, model, model_type),
                                         labels=(label,),
                                         num_samples=200,
                                         )
        storage.append(exp)
    return storage

def overall(model, model_type, text, class_names):
    """Given a model, the text and entity to predict and the class_names,
    
    Generates explanations for a prediction.
        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).
        Args:
            model: a machine learning model that is capable of predicting the class label of a given text.
            text: a string of text that needs to be classified.
            entity: a string of text indicating the entity to be classified.
            class_names: a list of strings representing the names of the classes that the model can predict, which has 
                been given the same order. For emotion, ["Sad", "Surprise", "Anger", "Disgust", "Neutral", "Fear", "Happy"]
                for stance, ["AGAINST", "NONE", "FAVOR]
            
        Returns:
            predictions: a one-dimensional NumPy array of predicted class labels for the input text. The length of this 
                array is equal to the number of classes that the model can predict.
            probas: a two-dimensional NumPy array of predicted probabilities for each class. The length of the first 
                dimension is the same as the length of the predictions array, while the length of the second dimension
                is the number of classes that the model can predict.
            lime_with_model_save_image(input,labels): a pyplot showing the LIME (Local Interpretable Model-Agnostic
                Explanations) visualization for the input text. The plot highlights the words in the input text that
                are most important in predicting the class label. This function takes in two arguments: input, which is 
                a list containing the text and entity strings, and labels, which is a list of integers representing the 
                class labels for which the LIME visualization is being generated. The output of this function is a 
                Matplotlib figure object that can be displayed or saved as an image.
        """
    if (type(model) == ClassificationModel):
        predictions, raw_outputs = model.predict(text)
    else:    
        texts = model.process_data(text)
        emotions, stance = model.predict(texts) 
        if model_type == 'Stance':
            raw_outputs = stance
        elif model_type == 'Emotion':
            raw_outputs = emotions           
        else:
            print('Error: Model has to be of type Stance or Emotions!')
            return
        predictions = raw_outputs.argmax(-1).numpy()
    probas = softmax(raw_outputs, axis =1)
    explainer = LimeTextExplainer(class_names = class_names)
    labels = [i for i in range(len(class_names))]
    return predictions, probas, lime_with_model_save_image_all(text[0], model_type, labels, explainer, model)

def zip_files(file_name, model_type, num_entries):
    zip_file = zipfile.ZipFile(file_name + "_{}_outputs.zip".format(model_type), "w")

    if model_type == 'Stance':
        if type(model) == ClassificationModel:
            labels = ['Favour', 'Neutral', 'Against']
        else:
            labels = ['Against', 'Favour', 'Neutral']
    elif model_type == 'Emotion':
        if type(model) == ClassificationModel:
            labels = ['Sadness', 'Surprise', 'Anger', 'Disgust', 'Neutral', 'Fear', 'Happy']
        else:
            labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise" ]
    
    for i in range(num_entries):
        current_file_name = '{}_{}_output_{}_graphical_prediction_probabilities.jpg'.format(file_name, model_type, i+1)
        zip_file.write(current_file_name, current_file_name)
        for label in labels:
            current_file_name = "{}_{}_output_{}_{}_graphical_explanation.jpg".format(file_name, model_type, i+1, label)
            zip_file.write(current_file_name, current_file_name)

    # Adding CSV Results file into zip folder
    zip_file.write(file_name + '_{}_results.csv'.format(model_type), file_name+ '_results.csv')

    zip_file.close()

    return zip_file

def create_results(model_type, model, csv_file_name, progress_placeholder):
    '''
    model_type -> String that is either "Stance" or "Emotion"
    model -> either stance or emotion model
    csv_file_name -> name of uploaded file containing list of texts to analyze
    - Header 1: Text - The full text that will be run through the model
    - Header 2: Named entity - The entity within the text which will have the classification applied
    '''

    if model_type == 'Stance':
        if type(model) == ClassificationModel:
            headers = ['Text', 'Named entity', 'Favour probability','Neutral probability', 'Against probability', 'Prediction']
            class_names = ['Favour','Neutral', 'Against']
        else:
            headers = ['Text', 'Named entity', 'Against probability','Favour probability', 'Neutral probability', 'Prediction']
            class_names = ['Against','Favour', 'Neutral']
    elif model_type == 'Emotion':
        if type(model) == ClassificationModel:
            headers = ['Text', 'Named entity', "Sadness probability", "Surprise probability", "Anger probability",
                    "Disgust probability", "Neutral probability", "Fear probability", "Happy probability", 'Prediction']
            class_names = ["Sadness", "Surprise", "Anger", "Disgust", "Neutral", "Fear", "Happy" ]
        else:
            headers = ['Text', 'Named entity', "Anger probability", "Disgust probability", "Fear probability",
                        "Happy probability", "Neutral probability", "Sadness probability", "Surprise probability", 'Prediction']
            class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise" ]
    else:
        raise Exception("Model Type has to be of either 'Stance' or 'Emotion'!")

    add_data_directory()

    query_df = csv_file_name
    num_entries = query_df.shape[0]
    storage = []

    with progress_placeholder.container(): # This will be overidden if replaced by another input for this placeholder
        st.subheader("Processing Explanation on Data Entries")
        progress_bar = st.progress(0.0, text="Processing data entries, kindly please wait. Completion Status: [{} out of {}]".format(0, num_entries))

    index_set = set()
    for index, row in tqdm(query_df.iterrows(), total = num_entries):
        text = [[row['Text'], row['Named entity']]]
        predictions, probas, images = overall(model, model_type, text, class_names)

        temp = text[0] + probas.tolist() + [class_names[predictions[0]]]
        output_row = []
        for element in temp:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    output_row.append(item)
            else:
                output_row.append(element)
        storage.append(output_row)
        pyplot_lime(csv_file_name.name, text, class_names, probas, model_type, index)
        for i, image in enumerate(images):
            current_file_name = "data/{}_{}_output_{}_{}_graphical_explanation.html".format(csv_file_name.name, model_type, index+1, headers[i+2][:-12])
            if index not in index_set:
                st.subheader("Graphical Explanation for Data Entry {} [File: {}]".format(index+1, csv_file_name.name))
                index_set.add(index)
            image.save_to_file(current_file_name, predict_proba = False)
            create_image_png(current_file_name)
            display_html(current_file_name, headers[i+2][:-12])
            progress_bar.progress((index+1)/num_entries, text="Processing data entries, kindly please wait. Completion Status: [{} out of {}]".format(index+1, num_entries))
    
    #Saving the results to path
    output_file_name = "data/" + csv_file_name.name

    results_df = pd.DataFrame(storage, columns = headers)
    results_df.to_csv(output_file_name + '_{}_results.csv'.format(model_type))

    # Preparing files to be downloaded for user
    zipped_file = zip_files(output_file_name, model_type, num_entries)
    download_explanation_graph(output_file_name + "_{}_outputs.zip".format(model_type))


### End of English Section ###

### Mandarin ###
# Mandarin text pre-processing #

def input_chinese_model_file():
    ml_model_file = st.file_uploader("Upload Mandarin ML Model Zip File", type=['zip'])
    return ml_model_file

def input_entities_chinese():
    ml_model_zh = input_chinese_model_file()
    data_file_zh = st.file_uploader("Upload Mandarin Prediction Data", type=['csv'])
    
    return ml_model_zh, data_file_zh

def load_tokenizer_chinese():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    return tokenizer

def load_ml_model_chinese(ml_model_file, model_type):
    with zipfile.ZipFile(ml_model_file) as z:
            z.extractall(".")
            # files = os.listdir(os.curdir + '/' + ml_model_file.name[:-4])
            # for file in files:
            #     st.write(file)

    try:
        import sys
        sys.path.insert(0, "./" + ml_model_file.name[:-4])
        from new.utils import get_model
        model_weight_path = "./" + ml_model_file.name[:-4] + '/' + ml_model_file.name[:-4] + '.pth'
        model = get_model(model_weight_path, 'zh')
    except:
        if model_type == 'Emotion':
            id2label = {0: "Sadness", 1: "Surprise", 2:"Anger", 3:'Disgust', 4:'Neutral', 5:'Fear', 6:'Happy'}
            label2id = {"Sadness":0, "Surprise":1, "Anger":2, 'Disgust':3, 'Neutral':4,'Fear':5,'Happy':6}
            model = AutoModelForSequenceClassification.from_pretrained("./" + ml_model_file.name[:-4], num_labels = 7,id2label=id2label, label2id=label2id )
        elif model_type == 'Stance':
            id2label = {0: "Against", 1: "Favour"}
            label2id = {"Against":0, "Favour":1}
            model = AutoModelForSequenceClassification.from_pretrained("./" + ml_model_file.name[:-4], num_labels = 2, id2label=id2label, label2id=label2id )
    return model

# def load_ml_model_chinese(ml_model_file, model_type):
#     with zipfile.ZipFile(ml_model_file) as z:
#             z.extractall(".")
#             files = os.listdir(os.curdir + '/' + ml_model_file.name[:-4])
#             for file in files:
#                 st.write(file)

    
#     import sys
#     sys.path.insert(0, "./" + ml_model_file.name[:-4])
#     from new.utils import get_model
#     model_weight_path = "./" + ml_model_file.name[:-4] + '/' + ml_model_file.name[:-4] + '.pth'
#     model = get_model(model_weight_path, 'zh')
#     return model
    
def segment_words(text):
    nlp_zh = spacy.load('zh_core_web_sm')
    string = ""
    words = nlp_zh(text)
    for token in words:
        string += token.text
        string += ' '
    return string[:-1]

def remove_spaces(string):
    return re.sub(" ", "", string)

def pass_entity_chinese(entity, model, model_type): 
    if type(model) == BertForSequenceClassification:
        def predictor(texts):
            tokenizer = load_tokenizer_chinese()
            #change back to normal string
            texts = list(map(lambda x: [remove_spaces(x), entity], texts))
            outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
            probas = F.softmax(outputs.logits).detach().numpy()
            return probas
    else:
        def predictor(texts):
            texts = list(map(lambda x: [remove_spaces(x),entity], texts))
            emotion, stance = model.predict(texts)
            if model_type == 'Stance':
                raw_outputs = stance
            elif model_type == 'Emotion':
                raw_outputs = emotion
            else:
                print('Error: Model has to be of type Stance or Emotion!')
                return
            #This is where model.predict is needed for each input -> returns the actual prediction AND probabilities of each class for individual input
            probas = softmax(raw_outputs, axis =1)
            return probas
    return predictor

def lime_with_model_save_image_chinese(input, model_type, labels, explainer, model):
    text, entity = input
    storage = []
    # Segment words here according to spacy
    segmented_text = segment_words(text)
    if type(model) == BertForSequenceClassification and model_type == 'Stance':
        exp = explainer.explain_instance(segmented_text,
                                            classifier_fn = pass_entity_chinese(entity, model, model_type),
                                            labels=labels,
                                            num_samples=200, 
                                            top_labels = 1)
        storage.append(exp)
    else:
        for label in labels:
            exp = explainer.explain_instance(segmented_text,
                                            classifier_fn = pass_entity_chinese(entity, model, model_type),
                                            labels=(label,),
                                            num_samples=200, 
                                            )
            storage.append(exp)         
    return storage

def overall_chinese(model, model_type, text, class_names):
    if (type(model) == BertForSequenceClassification):
        tokenizer = load_tokenizer_chinese()
        raw_outputs = model(**tokenizer(text, return_tensors="pt", padding=True))
        predictions = [argmax(raw_outputs.logits).item()]
        probas = F.softmax(raw_outputs.logits).detach().numpy()
    else:    
        texts = model.process_data(text)
        emotions, stance = model.predict(texts) 
        
        if model_type == 'Stance':
            raw_outputs = stance
            
        elif model_type == 'Emotion':
            raw_outputs = emotions
            
            #pyplot_lime(text, class_names, probas, model_type)
        else:
            print('Error: Model has to be of type Stance or Emotions!')
            return
        predictions = raw_outputs.argmax(-1).numpy()
        probas = softmax(raw_outputs, axis =1)
    explainer = LimeTextExplainer(class_names = class_names)
    labels = [i for i in range(len(class_names))]
    return predictions, probas, lime_with_model_save_image_chinese(text[0], model_type, labels, explainer, model)

def zip_files_chinese(file_name, model_type, num_entries, model):
    zip_file = zipfile.ZipFile(file_name + "_{}_outputs.zip".format(model_type), "w")

    if model_type == 'Stance':
        if type(model) == BertForSequenceClassification:
            labels = ['Stance']
        else:
            labels = ['Against', 'Favour', 'Neutral']
    elif model_type == 'Emotion':
        if type(model) == BertForSequenceClassification:
            labels = ['Sadness', 'Surprise', 'Anger', 'Disgust', 'Neutral', 'Fear', 'Happy']
        else:
            labels = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise" ]
    for i in range(num_entries):
        current_file_name = '{}_{}_output_{}_graphical_prediction_probabilities.jpg'.format(file_name, model_type, i+1)
        zip_file.write(current_file_name, current_file_name)
        for label in labels:
            current_file_name = "{}_{}_output_{}_{}_graphical_explanation.jpg".format(file_name, model_type, i+1, label)
            zip_file.write(current_file_name, current_file_name)

    # Adding CSV Results file into zip folder
    zip_file.write(file_name + '_{}_results.csv'.format(model_type), file_name+ '_results.csv')

    zip_file.close()

    return zip_file

def create_results_chinese(model_type, model, csv_file_name, progress_placeholder):
        if model_type == 'Stance':
            if type(model) == BertForSequenceClassification:
                #headers = ['Text', 'Named entity', 'Stance', 'Prediction']
                headers = ['Text', 'Named entity', 'Against probability', 'Favour probability', 'Prediction']
                class_names = ['Against', 'Favour']
            else:
                headers = ['Text', 'Named entity', 'Against probability','Favour probability', 'Neutral probability', 'Prediction']
                class_names = ['Against', 'Favour', 'Neutral']
        elif model_type == 'Emotion':
            if type(model) == BertForSequenceClassification:
                headers = ['Text', 'Named entity', "Sadness probability", "Surprise probability", "Anger probability",
                        "Disgust probability", "Neutral probability", "Fear probability", "Happy probability", 'Prediction']
                class_names = ["Sadness", "Surprise", "Anger", "Disgust", "Neutral", "Fear", "Happy" ]
            else:
                headers = ['Text', 'Named entity', "Anger probability", "Disgust probability", "Fear probability",
                        "Happy probability", "Neutral probability", "Sadness probability", "Surprise probability", 'Prediction']
                class_names = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sadness", "Surprise" ]
        else:
            raise Exception("Model Type has to be of either 'Stance' or 'Emotion'!")

        add_data_directory()

        query_df = csv_file_name
        num_entries = query_df.shape[0]
        storage = []

        with progress_placeholder.container(): # This will be overidden if replaced by another input for this placeholder
            st.subheader("Processing Explanation on Data Entries")
            progress_bar = st.progress(0.0, text="Processing data entries, kindly please wait. Completion Status: [{} out of {}]".format(0, num_entries))

        index_set = set()
        for index, row in tqdm(query_df.iterrows(), total = num_entries):
            # text, entity = row['Text'], row['Named entity']
            text = [[row['Text'], row['Named entity']]]

            predictions, probas, images = overall_chinese(model, model_type, text, class_names)

            temp = text[0] + probas.tolist() + [class_names[predictions[0]]]
            output_row = []
            for element in temp:
                if type(element) is list:
                    # If the element is of type list, iterate through the sublist
                    for item in element:
                        output_row.append(item)
                else:
                    output_row.append(element)
            storage.append(output_row)
            
            if type(model) == BertForSequenceClassification and model_type == 'Stance':
                
                pyplot_chinese_stance(csv_file_name.name, text, class_names, probas, model_type, index)
            else:
                pyplot_lime(csv_file_name.name, text, class_names, probas, model_type, index)
            for i, image in enumerate(images):
                if type(model) == BertForSequenceClassification and model_type == 'Stance':
                    current_file_name = "data/{}_{}_output_{}_{}_graphical_explanation.html".format(csv_file_name.name, model_type, index + 1, 'Stance')
                else:
                    current_file_name = "data/{}_{}_output_{}_{}_graphical_explanation.html".format(csv_file_name.name, model_type, index+1, headers[i+2][:-12])
                if index not in index_set:
                    st.subheader("Graphical Explanation for Data Entry {} [File: {}]".format(index+1, csv_file_name.name))
                    index_set.add(index)
                image.save_to_file(current_file_name, predict_proba = False)
                create_image_png(current_file_name)
                display_html(current_file_name, headers[i+2][:-12])
                progress_bar.progress((index+1)/num_entries, text="Processing data entries, kindly please wait. Completion Status: [{} out of {}]".format(index+1, num_entries))
        
        #Saving the results to path
        output_file_name = "data/" + csv_file_name.name

        results_df = pd.DataFrame(storage, columns = headers)
        results_df.to_csv(output_file_name + '_{}_results.csv'.format(model_type))

        # Preparing files to be downloaded for user
        zipped_file = zip_files_chinese(output_file_name, model_type, num_entries, model)
        download_explanation_graph(output_file_name + "_{}_outputs.zip".format(model_type))


### End of Mandarin Section ###

### Bahasa Melayu ###
def input_malay_model_file():
    ml_model_file = st.file_uploader("Upload Bahasa Melayu ML Model Zip File", type=['zip'])
    return ml_model_file

def input_entities_malay():
    ml_model_file = input_malay_model_file()
    data_file = st.file_uploader("Upload Bahasa Melayu Prediction Data", type=['csv'])
    
    return ml_model_file, data_file
    
def load_ml_model_malay(ml_model_file, model_type):

    with zipfile.ZipFile(ml_model_file) as z:
        z.extractall(".")
    model = ClassificationModel("bert", "./" + ml_model_file.name[:-4], use_cuda = False)
    return model

def pass_entity_malay(entity,model):
    def predictor(texts):
        texts = list(map(lambda x: [x, entity], texts))
        predictions, raw_outputs = model.predict(texts)
        probas = softmax(raw_outputs, axis =1)
        return probas
    return predictor

def lime_with_model_save_image_malay(input,labels,explainer,model):
    text, entity = input
    storage = []
    for label in labels:
        exp = explainer.explain_instance(text,
                                         classifier_fn = pass_entity_malay(entity, model),
                                         labels=(label,),
                                         num_samples=20,
                                         )
        storage.append(exp)
    return storage

def overall_malay(model,text,class_names):
    predictions,raw_outputs = model.predict(text)
    probas = softmax(raw_outputs, axis =1)
    explainer = LimeTextExplainer(class_names=class_names)
    labels = [i for i in range(len(class_names))]
    return predictions, probas, lime_with_model_save_image_malay(text[0],labels,explainer,model)

def zip_files_malay(file_name, model_type, num_entries):
    zip_file = zipfile.ZipFile(file_name + "_{}_outputs.zip".format(model_type), "w")

    if model_type == 'Stance':
        labels = ['Against', 'Neutral', 'Favour']
    if model_type == 'Emotion':
        labels = ['Sadness', 'Surprise', 'Anger', 'Fear', 'Happy', 'Love']
    
    for i in range(num_entries):
        current_file_name = '{}_{}_output_{}_graphical_prediction_probabilities.jpg'.format(file_name, model_type, i+1)
        zip_file.write(current_file_name, current_file_name)

        for label in labels:
            current_file_name = "{}_{}_output_{}_{}_graphical_explanation.jpg".format(file_name, model_type, i+1, label)
            zip_file.write(current_file_name, current_file_name)

    # Adding CSV Results file into zip folder
    zip_file.write(file_name + '_{}_results.csv'.format(model_type), file_name+ '_results.csv')

    zip_file.close()

    return zip_file

def create_results_malay(model_type, model, csv_file_name, progress_placeholder):
    '''
    model_type -> String that is either "Stance" or "Emotion"
    model -> either stance or emotion model
    csv_file_name -> name of uploaded file containing list of texts to analyze
    - Header 1: Text - The full text that will be run through the model
    - Header 2: Named entity - The entity within the text which will have the classification applied
    '''
    if model_type == 'Stance':
        headers = ['Text', 'Named entity', 'Against probability','Neutral probability', 'Favour probability', 'Prediction']
        class_names = ['Against', 'Neutral', 'Favour']
    elif model_type == 'Emotion':
        headers = ['Text', 'Named entity', "Sadness probability", "Surprise probability", "Anger probability",
                   "Fear probability", "Happy probability", "Love probability", 'Prediction']
        class_names = ["Sadness", "Surprise", "Anger", "Fear", "Happy", "Love"]
    else:
        raise Exception("Model Type has to be of either 'Stance' or 'Emotion'!")
    
    add_data_directory()

    query_df = csv_file_name
    num_entries = query_df.shape[0]
    storage = []

    with progress_placeholder.container(): # This will be overidden if replaced by another input for this placeholder
        st.subheader("Processing Explanation on Data Entries")
        progress_bar = st.progress(0.0, text="Processing data entries, kindly please wait. Completion Status: [{} out of {}]".format(0, num_entries))

    index_set = set()
    for index, row in tqdm(query_df.iterrows(), total = num_entries):
        text = [[row['Text'], row['Named entity']]]
        predictions, probas, images = overall_malay(model, text, class_names)
        #st.write(text)
        #st.write(predictions)
        temp = text[0] + probas.tolist() + [class_names[predictions[0]]]
        
        output_row = []
        for element in temp:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    output_row.append(item)
            else:
                output_row.append(element)
        storage.append(output_row)
        pyplot_lime(csv_file_name.name, text, class_names, probas, model_type, index)
        for i, image in enumerate(images):
            current_file_name = "data/{}_{}_output_{}_{}_graphical_explanation.html".format(csv_file_name.name, model_type, index+1, headers[i+2][:-12])
            if index not in index_set:
                st.subheader("Graphical Explanation for Data Entry {} [File: {}]".format(index+1, csv_file_name.name))
                index_set.add(index)
            image.save_to_file(current_file_name, predict_proba = False)
            create_image_png(current_file_name)
            display_html(current_file_name, headers[i+2][:-12])
            progress_bar.progress((index+1)/num_entries, text="Processing data entries, kindly please wait. Completion Status: [{} out of {}]".format(index+1, num_entries))
    
    #Saving the results to path
    output_file_name = "data/" + csv_file_name.name

    results_df = pd.DataFrame(storage, columns = headers)
    results_df.to_csv(output_file_name + '_{}_results.csv'.format(model_type))

    # Preparing files to be downloaded for user
    zipped_file = zip_files_malay(output_file_name, model_type, num_entries)
    download_explanation_graph(output_file_name + "_{}_outputs.zip".format(model_type))

### End of Bahasa Melayu Section ###

def pyplot_lime(file_name, text, class_names, probas, model_type, index):
    st.subheader("Graphical Prediction Probabilities for Data Entry {} [File: {}]".format(index+1, file_name))
    fig, ax = plt.subplots(figsize=(10, 2), dpi=600)
    x = class_names
    y = probas[0]
    mapping_emo = {'Sadness': 'skyblue', 'Surprise': 'orange', 'Anger': 'red', 'Disgust': 'green', 'Neutral': 'grey', 
                   'Fear': 'purple', 'Happy': 'yellow', 'Love': 'pink'}
    mapping_stance = {'Against': 'lightcoral', 'Neutral': 'grey', 'Favour': 'skyblue'}
    if model_type == 'Emotion':
        bars = ax.bar(x, y, color = [mapping_emo[x] for x in class_names])
        ax.set_xlabel('Emotion', fontsize = 8)
    elif model_type == 'Stance':
        bars = ax.bar(x, y, color = [mapping_stance[x] for x in class_names])
        ax.set_xlabel('Stance', fontsize = 8)
    ax.set_ylabel('Probability %', fontsize = 8)

    text_ner = [item for sublist in text for item in sublist]
    result = '; NER: '.join(text_ner)
    # add percentages above x-axis at y=0
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, 0, f'{y[i]/sum(y)*100:.0f}%', ha='center', va='bottom')

    # set x-tick positions and labels
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)

    # set y-axis limit to 120% of maximum value
    ax.set_ylim(top=max(y)*1.2)

    ax.tick_params(axis='both', labelsize = 8)
    ax.set_title("\n".join(textwrap.wrap(result, 100)), fontsize = 8)
    st.pyplot(fig)
    plt.savefig('data/{}_{}_output_{}_graphical_prediction_probabilities.jpg'.format(file_name, model_type, index+1))
    return

def pyplot_chinese_stance(file_name, text, class_names, probas, model_type, index):
    st.subheader("Graphical Prediction Probabilities for Data Entry {} [File: {}]".format(index+1, file_name))
    fig, ax = plt.subplots(figsize=(10, 2), dpi=600)
    x = class_names
    y = probas[0]
    bars = ax.bar(x, y, color = ["skyblue", "pink"])
    ax.set_xlabel('Stance', fontsize = 8)
    ax.set_ylabel('Probability %', fontsize = 8)
    text_ner = [item for sublist in text for item in sublist]
    result = '; NER:  '.join(text_ner)
    # add percentages above x-axis at y=0
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, 0, f'{y[i]/sum(y)*100:.0f}%', ha='center', va='bottom')

    # set x-tick positions and labels
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)

    # set y-axis limit to 120% of maximum value
    ax.set_ylim(top=max(y)*1.2)

    ax.tick_params(axis='both', labelsize = 8)
    ax.set_title("\n".join(textwrap.wrap(result, 100)), fontsize = 8)
    st.pyplot(fig)
    plt.savefig('data/{}_{}_output_{}_graphical_prediction_probabilities.jpg'.format(file_name, model_type, index+1))
    return

def parse_data_file(data_file):
    return data_file

def add_data_directory():
    cwd = os.getcwd()

    if not os.path.exists(cwd+'/data'):
        os.makedirs('data')


def create_image_png(file_name):
    hti = Html2Image(custom_flags=['--default-background-color=ffffff'], output_path='data')
    hti.screenshot(html_file=file_name, save_as='{}.jpg'.format(file_name[5:-5]))


def display_html(file_name, label):
    with open(file_name, 'r', encoding = 'UTF8') as f:
        html_data = f.read()
    
    #st.write(label + " Explanation:")
    st.components.v1.html(html_data, height=400)

def download_explanation_graph(downloadable_file):
    with open(downloadable_file, "rb") as file:
        image_download_button = st.download_button(
            label="Download All Outputs",
            data=file,
            file_name=downloadable_file,
            mime="application/zip"
        )

def footer(progress_placeholder):
    with progress_placeholder.container():
        st.header("Explanation Results on Data Entries")
        st.write("Task has successfully been completed! Scroll down to the bottom of the page to download outputs.")
    

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Langsplain: Multi-Lingual Text Classification Explainer")

    language, model_type = choose_language()
    input_type = st.selectbox("Data Input Type:", ('Multiple (as a .csv file)', 'Single (as a text input)'))


    if input_type == 'Multiple (as a .csv file)':

        if language == 'English':
            ml_model_file, data_file = input_entities()
            # try:
            #     df = pd.read_csv(data_file)
            #     df.dropna(inplace=True)
            #     df.name = str(data_file.name[:-4])
            #     st.write(df)
            # except:
            #     st.warning('Please upload a file.')

            if st.button("Run English Explanation Model!"):
                progress_placeholder = st.empty()
            
                if ml_model_file and data_file:
                    df = parse_data(data_file, input_type)
                    progress_placeholder.write("Loading Model...")
                    model = load_ml_model(ml_model_file)
                    create_results(model_type, model, df, progress_placeholder)
                else:
                    progress_placeholder.write("Missing input!")

                footer(progress_placeholder)

        elif language == 'Mandarin':
            ml_model_zh, data_file_zh = input_entities_chinese()

            if st.button("Run Mandarin Explanation Model!"):
                progress_placeholder = st.empty()
            
                if ml_model_zh and data_file_zh:
                    df = parse_data(data_file_zh, input_type)
                    progress_placeholder.write("Loading Model...")
                    model = load_ml_model_chinese(ml_model_zh, model_type)
                    create_results_chinese(model_type, model, df, progress_placeholder)
                else:
                    progress_placeholder.write("Missing input!")
                
                footer(progress_placeholder)


        elif language == 'Malay':
            ml_model_m, data_file_m = input_entities_malay()

            if st.button("Run Bahasa Melayu Explanation Model!"):
                progress_placeholder = st.empty()

                if ml_model_m and data_file_m:
                    df = parse_data(data_file_m, input_type)
                    progress_placeholder.write("Loading Model...")
                    model = load_ml_model_malay(ml_model_m, model_type)
                    create_results_malay(model_type, model, df, progress_placeholder)
                else:
                    progress_placeholder.write("Missing input!")
                
                footer(progress_placeholder)

    if input_type == 'Single (as a text input)':

        if language == 'English':
            ml_model_file = input_english_model_file()
            data = single_input_text()

            if st.button("Run English Explanation Model!"):
                progress_placeholder = st.empty()
            
                if ml_model_file and data:
                    df = parse_data(data, input_type)
                    progress_placeholder.write("Loading Model...")
                    model = load_ml_model(ml_model_file)
                    create_results(model_type, model, df, progress_placeholder)
                else:
                    progress_placeholder.write("Missing input!")
                
                footer(progress_placeholder)

        elif language == 'Mandarin':
            ml_model_zh = input_chinese_model_file()
            data = single_input_text()

            if st.button("Run Mandarin Explanation Model!"):
                progress_placeholder = st.empty()
            
                if ml_model_zh and data:
                    df = parse_data(data, input_type)
                    progress_placeholder.write("Loading Model...")
                    model = load_ml_model_chinese(ml_model_zh, model_type)
                    create_results_chinese(model_type, model, df, progress_placeholder)
                else:
                    progress_placeholder.write("Missing input!")
                
                footer(progress_placeholder)


        elif language == 'Malay':
            ml_model_m = input_malay_model_file()
            data = single_input_text()

            if st.button("Run Bahasa Melayu Explanation Model!"):
                progress_placeholder = st.empty()

                if ml_model_m:
                    df = parse_data(data, input_type)
                    #st.write(df)
                    progress_placeholder.write("Loading Model...")
                    model = load_ml_model_malay(ml_model_m, model_type)
                    create_results_malay(model_type, model, df, progress_placeholder)
                else:
                    progress_placeholder.write("Missing input!")
                
                footer(progress_placeholder)

    