# импорт библиотек
import sys
import numpy as np
from collections import Counter
import random
import math
import os
import functions
from functions import *


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


# выводим наш список vocab со всеми упомянутыми фонемами
print(vocab)
# длина списка vocab
print(len(vocab))
# выводим словарь, где каждой фонеме присвоен уникальный индекс (начиная с 0)
print(phon2index)
# длина словаря
print(len(phon2index))


'''Матрицы для предсказания'''
embed_size = 32 #размерность векторного представления фонем, то есть фонем из 50 элементов
learning_rate = 0.00000001 #устанавливаем скорость обучения
BATCH_SIZE = 4 #размер батча
EPOCHS = 50 #количество эпох


# Инициализируем веса с Xavier

'''Создаем матрицу эмбеддингов (векторных представлений) для фонем.
Инициализируем матрицу случайными значениями из нормального распределения со средним значением 0 и стандартным отклонением,
рассчитанным на основе размера словаря фонем размер матрицы = (количество фонем в словаре, размер эмбеддинга)'''
embed = np.random.normal(0, 1/np.sqrt(len(vocab)), size=(len(vocab), embed_size))

'''Создаем рекуррентную матрицу, инициализируем матрицу случайными значениями из нормального распределения со средним значением 0
и стандартным отклонением, рассчитанным на основе размера эмбеддинга размер матрицы = (размер эмбеддинга, размер эмбеддинга)'''
recurrent = np.random.normal(0, 1/np.sqrt(embed_size), size=(embed_size, embed_size))

'''Создаем начальный вектор скрытого состояния и инициализируем вектор нулями размер вектора = (размер эмбеддинга)'''
start = np.zeros(embed_size)

'''Создаем матрицу декодера. Инициализируем матрицу случайными значениями из нормального распределения со средним значением 0
и стандартным отклонением, рассчитанным на основе размера эмбеддинга размер матрицы = (размер эмбеддинга, количество фонем в словаре)'''
decoder = np.random.normal(0, 1/np.sqrt(embed_size), size=(embed_size, len(vocab)))

'''Создаем матрицу one-hot encoding. Создаем единичную матрицу, которая используется для представления фонем в виде one-hot вектора
размер матрицы = (количество фонем в словаре, количество фонем в словаре)'''
one_hot = np.eye(len(vocab))


print('Вектор эмбейдинга:', embed)
print('Матрица декодера:', decoder)


'''Обучение модели'''
# Путь для сохранения весов модели
save_path = "weights"


# Переменные для хранения лучших результатов
best_accuracy = 0.0
best_loss = float('inf')


# Разделение данных на обучающую и тестовую выборки 80/20
split_index = int(len(tokens) * 0.8)
train_tokens = tokens[:split_index]
test_tokens = tokens[split_index:]



'''Обучение модели'''
for epoch in range(EPOCHS):
    # создаем батчи из обучающих данных
    train_batches = create_batches(train_tokens, BATCH_SIZE, phons2indices=lambda word: phons2indices(word, phon2index))
    # задаем переменню общей ошибки, точности, слов
    total_loss = 0
    total_accuracy = 0
    total_words = 0

    for batch in train_batches: #цикл для бача на обучающей выборке (задаем все по нулям)
        batch_loss = 0
        batch_accuracy = 0
        batch_words = 0

        for word in batch: #цикл для каждого слова в батче
            '''Получаем результаты предсказания модели:
            layers: список слоев, содержащих предсказанные значения и другие данные
            loss: значение ошибки для данного слова
            accuracy: значение точности для данного слова'''
            layers, loss, accuracy = predict(word, start, recurrent, decoder, embed, one_hot)
            batch_loss += loss
            batch_accuracy += accuracy
            batch_words += 1

            # обратное распространение для обновления весов (вычисляем градиенты ошибки и используем их для корректировки весов модели)
            for layer_idx in reversed(range(len(layers))):
                # итерируем по слоям в обратном порядке
                layer = layers[layer_idx]
                # получаем текущий слой
                target = word[layer_idx-1]

                # проверяем, что слой не первый (входной)
                if(layer_idx > 0):
                    layer['output_delta'] = layer['pred'] - one_hot[target] #вычисляем разницу между предсказанным и ожидаемым вектором
                    new_hidden_delta = layer['output_delta'].dot(decoder.transpose()) #вычисляем ошибку скрытого состояния текущего слоя (транспонируя матрицу весов decoder и перемножая с дельтой выходного слоя)

                    # проверяем, что слой не последний (выходной)
                    if(layer_idx == len(layers)-1):
                        # устанавливаем ошибку скрытого слоя равной вычисленной ошибке
                        layer['hidden_delta'] = new_hidden_delta
                    else:
                        # ошибка скрытого состояния вычисляется на основе ошибки текущего слоя и ошибки скрытого состояния следующего слоя
                        layer['hidden_delta'] = new_hidden_delta + layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())
                else: #если слой первый
                    # вычисляем ошибку скрытого состояния первого слоя на основе ошибки скрытого состояния второго слоя
                    layer['hidden_delta'] = layers[layer_idx+1]['hidden_delta'].dot(recurrent.transpose())

            # Ограничение градиента (для предотвращения градиентного взрыва)
            for layer in layers: #цикл по слоям
                if 'hidden_delta' in layer: #проверяем, есть ли ошибка скрытого слоя в текущем слое
                    grad_norm = np.linalg.norm(layer['hidden_delta']) # вычисляем норму градиента
                    if grad_norm > 10: #устанавливаем максимальную норму
                        layer['hidden_delta'] = layer['hidden_delta'] * (10 / grad_norm) # нормализуем градиент если он большой

            '''теперь корректируем веса'''
            # обновляем начальное скрытое состояние на основе ошибки скрытого состояния первого слоя, деля на длину предложения для нормализации
            # вычитаем из начального скрытого состояния (start) произведение ошибки скрытого состояния первого слоя, скорости обучения и деленное на длину предложения.
            start -= layers[0]['hidden_delta'] * learning_rate / (len(word))
            # итерация по всем слоям со второго
            for layer_idx,layer in enumerate(layers[1:]):
                # корректируем матрицу декодера с учетом скрытого состояния текущего слоя и ошибки выходного слоя, деля на длину слова для нормализации
                decoder -= np.outer(layers[layer_idx]['hidden'], layer['output_delta']) * learning_rate / (len(word))
                # извлекаем индекс текущего слова из списка индексов предложения
                embed_idx = word[layer_idx]
                # обновляем вектор эмбеддинга текущего слова (embed[embed_idx]), вычитая из текущего значения вектора произведение ошибки скрытого состояния, скорости обучения и деленное на длину слова
                # корректирует вектор эмбеддинга для уменьшения ошибки
                embed[embed_idx] -= layers[layer_idx]['hidden_delta'] * learning_rate / (len(word)) #корректируем вектор эмбеддинга текущего слова с помощью ошибки скрытого состояния текущего слоя, деля на длину предложения для нормализации
                # обновляем матрицу рекуррентной связи, вычитая из текущего значения матрицы внешнее произведение скрытого состояния и ошибки скрытого состояния, умноженное на скорость обучения и деленное на длину предложения
                # корректирует матрицу рекуррентной связи для уменьшения ошибки при обновлении скрытого состояния
                recurrent -= np.outer(layers[layer_idx]['hidden'], layer['hidden_delta']) * learning_rate / (len(word)) #обновляем матрицу рекуррентной связи с помощью скрытого состояния текущего слоя и ошибки скрытого состояния текущего слоя, деля на длину предложения для нормализации

        if batch_words > 0: #проверка на пустые батчи
            total_loss += batch_loss #суммируем ошибки для всего батча
            total_accuracy += batch_accuracy #суммируем точность для всего батча
            total_words += batch_words #суммируем количество слов в батче
    if total_words > 0: #проверяем было ли обработано хоть одно слово в текущей эпохе
         avg_loss = total_loss/total_words #среднее значение ошибки
         avg_accuracy = total_accuracy/total_words #среднее значение точности
    else: #если в эпохе не было обработано слов, то средние значения равны нулю
         avg_loss = 0
         avg_accuracy = 0

    # оценка модели на тестовой выборке
    test_accuracy = 0
    test_words = 0
    test_loss = 0

    # создаем батчи для тестовой выборки
    test_batches = create_batches(test_tokens, BATCH_SIZE, phons2indices=lambda word: phons2indices(word, phon2index))
    # цикл по тестовым батчам
    for test_batch in test_batches:
        batch_test_accuracy = 0 #переменная для хранения точности текущего тестового батча
        batch_test_loss = 0 #переменная для хранения потери текущего тестового батча
        batch_test_words = 0 #переменняа для хранения количсетва слов в тестовом батче

        # цикл по всем словам в тестовом батче
        for word in test_batch:
            layers, loss, accuracy = predict(word, start, recurrent, decoder, embed, one_hot) #получаем предсказания модели, ошибку и точность для текущего слова
            batch_test_accuracy += accuracy #добавляем точность
            batch_test_loss += loss #добавляем потерю
            batch_test_words += 1 #бобавляем счет слов

        # проверка на наличие хотя бы одного слова в батче и так же суммируем
        if batch_test_words > 0:
            test_accuracy += batch_test_accuracy
            test_loss += batch_test_loss
            test_words += batch_test_words

    # проверяем, было ли обработано хотя бы одно слово в тестовой выборке
    if test_words > 0:
        avg_test_accuracy = test_accuracy/test_words #вычисляем среднюю точность, деля тестовую точность на количество слов
        avg_test_loss = test_loss/test_words #вычисляем среднюю ошибку, деля тестовую ошибку на количество слов
    # если не было обработано ни одного слова средние значения 0
    else:
        avg_test_accuracy = 0
        avg_test_loss = 0

    # вывод процесса обучения
    print(f"Эпоха {epoch + 1}/{EPOCHS}, "
          f"Обучение: Loss: {avg_loss:.3f}, Accuracy: {avg_accuracy:.3f} , "
          f"Тест: Loss: {avg_test_loss:.3f}, Accuracy: {avg_test_accuracy:.3f}")
    # прерываем обучение если среднняя ошибка nan
    if np.isnan(avg_loss):
        print('Ошибка: nan')
        break

    # проверка на улучшение
    # ниже мы проверяем улучшилась ли модель от эпохи к эпохе
    # устанавливая условие - либо текущая точность на тестовой выборке (avg_test_accuracy) больше лучшей предыдущей (best_accuracy),
    # либо текущая точность равна лучшей, но текущая ошибка на тестовой выборке (avg_test_loss) меньше лучшей предыдущей (best_loss)
    if avg_test_accuracy > best_accuracy or (avg_test_accuracy == best_accuracy and avg_test_loss < best_loss):
            # обновляем лучшую точность на ту, которую получили по условию
            best_accuracy = avg_test_accuracy
            # так же с потерей
            best_loss = avg_test_loss
            # сохранение весов в нашу папку для весов
            save_model(save_path, embed, recurrent, start, decoder)


# тестирование модели
word_index = 5
# получаем предсказания модели для слова с индексом word_index, преобразуя его фонемы в числовые индексы
l, _, _ = predict(phons2indices(tokens[word_index], phon2index), start, recurrent, decoder, embed, one_hot)
print(tokens[word_index])
# цикл по всем слоям в списке предсказаний (l), кроме входного и выходного слоя
for i, each_layer in enumerate(l[1:-1]):
    # получаем входную фонему для текущего слоя из оригинального слова
    input_phon = tokens[word_index][i]
    # получаем настоящую фонему для текущего слоя из оригинального слова
    true_phon = tokens[word_index][i + 1]
    # получаем предсказаную фонему для текущего слоя из оригинального слова (argmax() используется для нахождения индекса максимальной вероятности в предсказании)
    pred_phon = vocab[each_layer['pred'].argmax()]
    print(f"Предыдущая фонема: {input_phon} Истинная: {true_phon} Предсказанная: {pred_phon}")


word_index = 15
l, _, _ = predict(phons2indices(tokens[word_index], phon2index), start, recurrent, decoder, embed, one_hot)
print(tokens[word_index])
# цикл по всем слоям в списке предсказаний (l), кроме входного и выходного слоя
for i, each_layer in enumerate(l[1:-1]):
    # получаем входную фонему для текущего слоя из оригинального слова
    input_phon = tokens[word_index][i]
    # получаем настоящую фонему для текущего слоя из оригинального слова
    true_phon = tokens[word_index][i + 1]
    # получаем предсказаную фонему для текущего слоя из оригинального слова
    pred_phon = vocab[each_layer['pred'].argmax()]
    print(f"Предыдущая фонема: {input_phon} Истинная: {true_phon} Предсказанная: {pred_phon}")
