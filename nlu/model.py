import yaml
import numpy as np
import tensorflow as tf
import tensorflow.keras.models import Sequential
import tensorflow.keras.layers import LSTM, Dense
import tensorflow.keras.utils import to_categorical


data = yaml.safe_load(open('nlu\\train.yml', 'r', encoding='utf-8').read())

inputs, outputs = [], []

for command in data['commands']:
    inputs.append(command['input'].lower())
    outputs.append('{}\{}'.format(command['entity'], command['action']))

# Processar texto: palavras, caractéres, bytes, sub-palavras

chars = set()

for input in inputs + outputs:
    for ch in input:
        if ch not in chars:
            chars.add(ch)

# Mapear char-idx

chr2idx = {}
idx2chr = {}

for i, ch in enumerate(chars):
    chr2idx[ch] = i
    idx2chr[i] = ch

max_seq = max([len(x) for x in inputs])

print('Números de chars:', len(chars))
print('Maior seq:', max_seq)

# Criar o dataset one-hot (número de exemplos, tamanho de seq, num caracteres) 
# Criar data set disperso (número de exemplos, tamanho de seq)

input_data = np.zeros((len(inputs), max_seq, len(chars)), dtype='int32')
output_data = to_categorical(outputs, len(outputs))

for i, input in enumerate(inputs):
    for k, ch in enumerate(input):
        input_data[i, k, chr2idx[ch]] = 1.0

print(input_data[4])

'''
print(inputs)
print(outputs)
'''
