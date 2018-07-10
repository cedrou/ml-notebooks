#
# this is a reimplementation of the Sample from CNTK in Keras
#   CNTK-Samples-2-5-1\Examples\Text\CharacterLM
#

import numpy as np
import os
import sys

# force device before importing keras
from cntk import device
device.try_set_default_device(device.gpu(0))

from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Dense, Input, LSTM
from keras.optimizers import SGD
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.utils import to_categorical


rootpath = os.path.dirname(os.path.abspath(__file__))

print('Loading the data...')
dataset = "julesverne"

# load the full dataset file as text
datapath = os.path.join(rootpath, f'data/{dataset}.txt')
with open(datapath, 'r', encoding='utf8') as df:
    text = df.read().lower()

#text = text[:1024]

# create a sorted list of all characters in the dataset, and the maps to pass from chars to index in the list.
chars = sorted(list(set(text)))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# write vocab in a file for future use
with open(f'{datapath}.vocab', 'w', encoding='utf8') as vf:
    for c in chars:
        vf.write(f'{c}\n') if c != '\n' else vf.write('\n')

text_size, vocab_size = len(text), len(chars)
print(f'corpus size: {text_size}')
print(f'vocabulary size: {vocab_size}')

batch_size = maxlen = 100
def batch_generator():
    current = 0
    while True:
        x = []
        y = []
        for i in range(current, current + maxlen):
            sentence = [char_to_ix[c] for c in text[i: i + maxlen]]
            next_char = char_to_ix[text[i + maxlen]]

            x.append(to_categorical(sentence, vocab_size))
            y.append(to_categorical(next_char, vocab_size))

        current = (current + maxlen) % (text_size - 2*maxlen)
        yield np.array(x), np.array(y)



# Creates a character-level language model
initial_epoch = 0

def create_model():
    print('Create the Language Model...')
    model = Sequential()
    model.add(BatchNormalization(input_shape=(maxlen, vocab_size,)))
    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))
    model.summary()

    # Instantiate the optimizer object to drive the model training
    optimizer = SGD(lr=0.001, momentum=0.9990913221888589, clipnorm=5.0)

    #compile the model 
    model.compile(optimizer, loss='categorical_crossentropy')

    return model

def load_checkpoint(epoch):
    chkpath = os.path.join(rootpath, f'checkpoints/{dataset}.{epoch:02d}.hdf5')
    print(f'Load the Language Model from {chkpath}...')
    model = load_model(chkpath)
    model.summary()

    return model


# Select here either to train a new model, or to resume a previous training run

#model = create_model()

model = load_checkpoint(9)
initial_epoch = 9



# Sample from the network
def sample(preds, use_hardmax=True, temperature=1.0):
    p = preds
    if use_hardmax:
        w = np.argmax(p)
    else:
        # apply temperature: T < 1 means smoother; T=1.0 means same; T > 1 means more peaked
        p = np.power(p, temperature)
        # normalize
        p = p / np.sum(p)
        w = np.random.choice(range(vocab_size), p=p)
    
    return w


# Generate text from a seed sentence
from termcolor import colored
def generate(seed, diversity, size=100):
    
    #to_print = colored(seed, attrs=['reverse'])
    sys.stdout.write(seed)

    seed = list(seed)
    seed_len = len(seed)


    for _ in range(size):
        sentence = [char_to_ix[c] for c in seed[-seed_len:]]
        x_pred = np.array([to_categorical(sentence, vocab_size)])

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity==0, diversity)
        next_char = ix_to_char[next_index]

        seed += next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()

    print()

    return ''.join(seed)



# Function invoked at end of each batch. Prints generated text every 1000 batches
def on_batch_end(batch, logs):
    if batch % 1000 == 0:
        start_index = np.random.randint(0, text_size - maxlen - 1)
        seed = text[start_index: start_index + maxlen]
        #seed = ' '*(maxlen-1)+ 't'
        print(f'Seed: {seed}')
        for diversity in [0.0, 0.5, 1.0, 1.5]:
            print(f'----- diversity: {diversity} -----')
            generate(seed, diversity)



history = model.fit_generator(
    batch_generator(),
    steps_per_epoch = text_size // batch_size,
    initial_epoch=initial_epoch,
    epochs=50,
    callbacks=[
        LambdaCallback(on_batch_end=on_batch_end),
        ModelCheckpoint(f'{dataset}.{{epoch:02d}}.hdf5')
    ]
)

print ('Generate text')
seed = ' '*(maxlen-1) + np.random.choice(range(vocab_dim))
generate(seed, 1.5, 1000)

