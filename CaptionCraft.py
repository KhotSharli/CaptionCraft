#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Modules
import os
import pickle
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


# In[2]:


BASE_DIR = r"C:\Users\Shirley\Desktop\VGG16\Flickr8k"
WORKING_DIR = r"C:\Users\Shirley\Desktop\VGG16"


# In[3]:


#Extract Image Features

# load vgg16 model
model = VGG16()
# restructure the model
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)



# In[4]:


# extract features from image
#features = {}
#directory = os.path.join(BASE_DIR, 'Images')

#for img_name in tqdm(os.listdir(directory)):
    # load the image from file
#    img_path = directory + '/' + img_name
#    image = load_img(img_path, target_size=(224, 224))
    # convert image pixels to numpy array
#    image = img_to_array(image)
    # reshape data for model
#    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # preprocess image for vgg
#    image = preprocess_input(image)
    # extract features
#    feature = model.predict(image, verbose=0)
    # get image ID
#    image_id = img_name.split('.')[0]
    # store feature
#    features[image_id] = feature

# In[5]:


# store features in pickle
#pickle.dump(features, open(os.path.join(WORKING_DIR, 'features.pkl'), 'wb'))


# In[6]:


# load features from pickle
with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
    features = pickle.load(f)


# In[8]:


#Load the Captions Data

with open(os.path.join(BASE_DIR, 'Captions\\captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()


# In[9]:


# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)


# In[10]:


len(mapping)


# In[11]:


#Preprocess Text Data

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption


# In[12]:


# before preprocess of text
mapping['1000268201_693b08cb0e']


# In[13]:


# preprocess the text
clean(mapping)


# In[14]:


# after preprocess of text
mapping['1000268201_693b08cb0e']


# In[15]:


all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)


# In[16]:


len(all_captions)


# In[17]:


all_captions[:10]


# In[18]:


# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1


# In[19]:


vocab_size


# In[20]:


# get maximum length of the caption available
max_length = max(len(caption.split()) for caption in all_captions)
max_length


# In[21]:


#Train Test Split

image_ids = list(mapping.keys())
split = int(len(image_ids) * 0.90)
train = image_ids[:split]
test = image_ids[split:]


# In[22]:


# startseq girl going into wooden building endseq
#        X                   y
# startseq                   girl
# startseq girl              going
# startseq girl going        into
# ...........
# startseq girl going into wooden building      endseq


# In[23]:


# create data generator to get data in batch (avoids session crash)
def data_generator(data_keys, mapping, features, tokenizer, max_length, vocab_size, batch_size):
    # loop over images
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = mapping[key]
            # process each caption
            for caption in captions:
                # encode the sequence
                seq = tokenizer.texts_to_sequences([caption])[0]
                # split the sequence into X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pairs
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # store the sequences
                    X1.append(features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield [X1, X2], y
                X1, X2, y = list(), list(), list()
                n = 0


# In[24]:


#Model Creation

# encoder model
# image feature layers
inputs1 = Input(shape=(4096,))
fe1 = Dropout(0.4)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
# sequence feature layers
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.4)(se1)
se3 = LSTM(256)(se2)

# decoder model
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# plot the model
plot_model(model, show_shapes=True)


# In[25]:


# train the model
#epochs = 50
#batch_size = 32
#steps = len(train) // batch_size

#for i in range(epochs):
    # create data generator
    #generator = data_generator(train, mapping, features, tokenizer, max_length, vocab_size, batch_size)
    # fit for one epoch
    #model.fit(generator, epochs=1, steps_per_epoch=steps, verbose=1)


# In[26]:


# save the model
#model.save(WORKING_DIR+'/best_model.h5')
from keras.models import load_model

# Load the model
model = load_model(os.path.join(WORKING_DIR, 'best_model.h5'))

# In[27]:


#Generate Captions for the Image

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# In[28]:


# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text


# In[29]:


from nltk.translate.bleu_score import corpus_bleu
# validate with test data
actual, predicted = list(), list()

for key in tqdm(test):
    # get actual caption
    captions = mapping[key]
    # predict the caption for image
    y_pred = predict_caption(model, features[key], tokenizer, max_length) 
    # split into words
    actual_captions = [caption.split() for caption in captions]
    y_pred = y_pred.split()
    # append to the list
    actual.append(actual_captions)
    predicted.append(y_pred)
    
# calcuate BLEU score
print("BLEU-1: %f" % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))


# In[30]:


#Visualize the Results

from PIL import Image
import matplotlib.pyplot as plt
def generate_caption(image_name):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('.')[0]
    img_path = os.path.join(BASE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = mapping[image_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(y_pred)
    plt.imshow(image)


# In[31]:


generate_caption("1001773457_577c3a7d70.jpg")


# In[32]:


generate_caption("1002674143_1b742ab4b8.jpg")


# In[33]:


generate_caption("101669240_b2d3e7f17b.jpg")


# In[43]:


def generate_captions_for_image(image_path):
    # Load the pre-trained VGG16 model
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    
    # Load features from pickle (Assuming 'features.pkl' contains image features)
    with open(os.path.join(WORKING_DIR, 'features.pkl'), 'rb') as f:
        features = pickle.load(f)

    # Load tokenizer, max_length, vocab_size, and other necessary components
    
    # Load and preprocess the image for caption generation
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    # Extract image features
    image_features = vgg_model.predict(image)
    
    # Generate captions for the image
    # Implement the logic for caption generation using your trained model
    generated_caption = predict_caption(model, image_features, tokenizer, max_length)
    
    return generated_caption







# %%
