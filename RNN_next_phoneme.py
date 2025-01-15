'''
Обучение рекуррентной нейронной сети для предсказания следующей фонемы.
В данном коде мы будем обучать базовую рекуррентную нейронную сеть с прямым и обратным распространением

'''

# импорт библиотек
import sys
import numpy as np 
from collections import Counter
import random
import math


# чтение датасета, состоящего из фонетически транскрибированных слов (каждая фонема через пробел)
f = open('phon.txt','r')
raw = f.readlines()
f.close()


# создание пустого списка для токенов
tokens = list()
for line in raw[0:]:  
    tokens.append(line.lower().replace("\n","").split(" ")[0:]) #добавление слов в список токенов


# таким образом токены состоят из списков, содержащих слово в транскрипции 
print(tokens[0:5]) #вывод первых пяти элементов (токенов)
print(len(tokens)) #длина списка токенов


'''Формируем список существующих элементов из токенов и присваиваем им уникальный индекс'''
vocab = set() #создается простой список фонем из словаря
for sound in tokens: #итерация по каждому элементу в списке 'tokens'
    for phon in sound: #теперь по каждому звуку
        if phon == '': #проверям не пуста ли фонема, если да пропускаем
            pass
        else:
            vocab.add(phon) #добавляем фонему в 'vocab', если фонема уже есть, добавление не происходит
vocab = list(vocab) #преобразование множества в список
phon2index = {} #создание пустого словаря для хранения фонемы и ее индекса
for i,phon in enumerate(vocab):  #итерация по списку 'vocab', получая индекс i и фонему 'phon' 
    phon2index[phon]=i
    

def phons2indices(word): 
    '''Функция для преобразования списка Фонем в список индексов'''
    idx = list() #список для индексов
    for phon in word: #проходим по фонемам в слове
        idx.append(phon2index[phon]) #ищем индекс фонемы в словаре 'phon2index' и добавляет его в список 'idx'
    return idx


def softmax(x): 
    '''Функция активации softmax для предсказания следующей фонемы'''
    e_x = np.exp(x - np.max(x)) #ищем экспоненту каждого элемента вектора, вычитая максимальное значение 
    return e_x / e_x.sum(axis=0) #нормализуем экспоненциальные значения, деля на их сумму
    # возвращает вектор вероятностей


# выводим наш список vocab со всеми упомянутыми фонемами
print(vocab)
# длина списка vocab
print(len(vocab))
# выводим словарь, где каждой фонеме присвоен уникальный индекс (начиная с 0)
print(phon2index)
# длина словаря
print(len(phon2index))


'''Матрицы для предсказания'''
embed_size = 10 #размерность векторного представления фонем, то есть фонем из 50 элементов
embed = (np.random.rand(len(vocab),embed_size)) #векторов представлены случайными числами в диапазоне от 0 до 1 с плавающей точкой
recurrent = np.eye(embed_size) #рекуррентная матрица (первоначально единичная)
start = np.zeros(embed_size) #начальное векторное представление фонемы именно так начинается моделирование предлопредсказаний в нейронных сетях
decoder = (np.random.rand(embed_size, len(vocab))) #создается весовая матрица decoder для преобразования векторного представления фонемы в вектор вероятностей
one_hot = np.eye(len(vocab)) #вспомогательная (единичная) матрица для расчета функции потерь


print('Вектор эмбейдинга:', embed)
print('Матрица декодера:', decoder)




def predict(word):
    '''Функция принимает на вход список индексов фонем (слова)'''    
    layers = list() #Список с именем layers для прямого распространения
    layer = {} #словарб для слоёв
    layer['hidden'] = start #скрытое состояние первого слоя начальным вектором 'start'
    layers.append(layer) #добавляем первый слой в список
    loss = 0 #для хранения суммарной ошибки


    # итерация по слову
    for target_i in range(len(word)): #проходим по всем фонемам в слове
        layer = {} #словарь для этого слоя 
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder)) #извлекает вероятность, которую сеть предсказала для правильного следующего слова  
        # вычисляем ошибку для текущего предсказания, используя отрицательный логарифм вероятности правильного слова, добавляем ошибку к общей суммарной ошибке 'loss'
        epsilon = 1e-15 #добавляем маленькое число, которое используется для предотвращения взятия логарифма от нуля
        loss += -np.log(layer['pred'][word[target_i]] + epsilon) #чем ниже вероятность предсказанного слова, тем выше значение loss
        # выисляем скрытое состояние для следующего слоя, используя рекуррентную связь и эмбеддинги текущего слова
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[word[target_i]]
        layers.append(layer)
    return layers, loss #возвращаем список слоев 'layers' и суммарную ошибку 'loss'


'''Обучение модели'''
for iter in range(250000):
    # прямое распространение
    alpha = 0.0001 #сокрость обучения
    phon_tokens = tokens[iter%len(tokens)]
    phon_tokens = [token for token in phon_tokens if token] #отфильтровываем пустые строки
    word = phons2indices(phon_tokens) #выбираем случайное предложение из списка токенов и преобразуем его в список индексов
    layers,loss = predict(word) #получаем список слоев и ошибку


    # обратное распространение для обновления весов 
    for layer_idx in reversed(range(len(layers))):
        layer = layers[layer_idx] 
        target = word[layer_idx-1]
        # проверяем, что слой не первый 
        if(layer_idx > 0): 
            layer['output_delta'] = layer['pred'] - one_hot[target] #вычисляем разницу между предсказанным и ожидаемым вектором
            new_hidden_delta = layer['output_delta'].dot(decoder.transpose()) #вычисляем ошибку скрытого состояния текущего слоя


            # проверяем, что слой не последний
            if(layer_idx == len(layers)-1):
                layer['hidden_delta'] = new_hidden_delta
            else:
                # ошибка скрытого состояния вычисляется на основе ошибки текущего слоя и ошибки скрытого состояния следующего слоя
                layer['hidden_delta'] = new_hidden_delta + layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())
        else: #если слой первый 
            # вычисляем ошибку скрытого состояния первого слоя на основе ошибки скрытого состояния второго слоя
            layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())


    # теперь корректируем веса
    # обновляем начальное скрытое состояние на основе ошибки скрытого состояния первого слоя, деля на длину предложения для нормализации
    start -= layers[0]['hidden_delta'] * alpha / (len(word))
    # итерация по всем слоям со второго 
    for layer_idx,layer in enumerate(layers[1:]):
        # корректируем матрицу декодера с учетом скрытого состояния текущего слоя и ошибки выходного слоя, деля на длину предложения для нормализации
        decoder -= np.outer(layers[layer_idx]['hidden'], layer['output_delta']) * alpha / (len(word))
                # извлекаем индекс текущего слова из списка индексов предложения
        embed_idx = word[layer_idx]
        embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * alpha / (len(word)) #корректируем вектор эмбеддинга текущего слова с помощью ошибки скрытого состояния текущего слоя, деля на длину предложения для нормализации
        recurrent -= np.outer(layers[layer_idx]['hidden'], layer['hidden_delta']) * alpha / (len(word)) #обновляем матрицу рекуррентной связи с помощью скрытого состояния текущего слоя и ошибки скрытого состояния текущего слоя, деля на длину предложения для нормализации
    

    # выводим перплексию каждые 1000 итераций 
    if(iter % 10000 == 0):
        if str(np.exp(loss/len(word))) == 'nan':
               break
        else:
            print("Perplexity:", iter, (np.exp(loss/len(word))))

# посмотрим предсказание слова с индексом 5
word_index = 5
#  l - список слоев, _ - игнорируемая переменная loss
l,_ = predict(phons2indices(tokens[word_index]))
# выводим предложение с индексом 10 в виде списка
print(tokens[word_index])
# итерация по слоям кроме первого и последнего
for i,each_layer in enumerate(l[1:-1]):
    # извлекаем слово из предложения
    input = tokens[word_index][i]
    # извлекаем оригинальное следующее слово предложения
    true = tokens[word_index][i+1]
    # предсказанное следующее слово
    pred = vocab[each_layer['pred'].argmax()]
    # выводим результат
    print("Prev Input:" + input + (' ' * (12 - len(input))) +\
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)
    

# и на слово с индексом 15
word_index = 15
#  l - список слоев, _ - игнорируемая переменная loss
l,_ = predict(phons2indices(tokens[word_index]))
# выводим предложение с индексом 10 в виде списка
print(tokens[word_index])
# итерация по слоям кроме первого и последнего
for i,each_layer in enumerate(l[1:-1]):
    # извлекаем слово из предложения
    input = tokens[word_index][i]
    # извлекаем оригинальное следующее слово предложения
    true = tokens[word_index][i+1]
    # предсказанное следующее слово
    pred = vocab[each_layer['pred'].argmax()]
    # выводим результат
    print("Prev Input:" + input + (' ' * (12 - len(input))) +\
          "True:" + true + (" " * (15 - len(true))) + "Pred:" + pred)