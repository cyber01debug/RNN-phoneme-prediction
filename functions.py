import numpy as np
import os

def phons2indices(word, phon2index):
    '''Функция для преобразования списка Фонем в список индексов'''
    idx = list() #список для индексов
    for phon in word: #проходим по фонемам в слове
        if phon == '': #проверяем на пустую строчку
            pass
        else:
            idx.append(phon2index[phon]) #ищем индекс фонемы в словаре 'phon2index' и добавляет его в список 'idx'
    return idx

def softmax(x):
    '''Функция активации softmax для предсказания следующей фонемы'''
    e_x = np.exp(x - np.max(x)) #ищем экспоненту каждого элемента вектора, вычитая максимальное значение
    return e_x / e_x.sum(axis=0) #нормализуем экспоненциальные значения, деля на их сумму
    # возвращает вектор вероятностей

def create_batches(tokens, batch_size, phons2indices):
    '''Функция принимает список токенов (слов) и размер батча'''
    batches = [] #пустой список для батчей
    for i in range(0, len(tokens), batch_size): #итерирация по списку токенов с шагом, равным размеру батча
        batch = tokens[i:i + batch_size] #извлекаем из списка токена батч
        # фильтруем пустые слова и преобразуем в индексы
        batch = [phons2indices(word) for word in batch if word]
        # фильтруем пустые списки (слова)
        batch = [word for word in batch if word]
        if batch: #проверяем, что после фильтрации батч не пуст
            batches.append(batch) #добавляем батч
    return batches


def predict(word, start, recurrent, decoder, embed, one_hot):
    '''Функция принимает на вход список индексов фонем (слова)'''
    layers = list() #Список с именем layers для прямого распространения
    layer = {} #словарб для слоёв
    layer['hidden'] = start #скрытое состояние первого слоя начальным вектором 'start'
    layers.append(layer) #добавляем первый слой в список
    loss = 0 #для хранения суммарной ошибки
    correct_predictions = 0 #правильные предсказания
    total_predictions = 0 #все предсказания

    # итерация по слову
    for target_i in range(len(word)): #проходим по всем фонемам в слове
        layer = {} #словарь для этого слоя
        layer['pred'] = softmax(layers[-1]['hidden'].dot(decoder)) #извлекает вероятность, которую сеть предсказала для следующей фонемы

        # вычисляем ошибку для текущего предсказания, используя отрицательный логарифм вероятности правильного слова, добавляем ошибку к общей суммарной ошибке 'loss'
        # epsilon = 1e-15 #можно добавить маленькое число, которое используется для предотвращения взятия логарифма от нуля
        loss += -np.log(layer['pred'][word[target_i]]) #чем ниже вероятность предсказанного слова, тем выше значение loss
        layer['hidden'] = layers[-1]['hidden'].dot(recurrent) + embed[word[target_i]]
        layers.append(layer)

        predicted_index = np.argmax(layer['pred']) #предсказываем индекс следующей фонемы
        if predicted_index == word[target_i]: #если предсказанная фонема совпадает с правильной, то увеличиваем количество верных предсказаний
            correct_predictions += 1
        total_predictions += 1 #увеличиваем общее количество предсказаний

    # вычисляем точность модели
    # делим количество верных предсказаний на общее количество предсказаний
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    # возвращаем результат (слои потерю и точность)
    return layers, loss, accuracy


def save_model(save_path, embed, recurrent, start, decoder):
    """Функция для сохранения весов модели"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, "embed.npy"), embed)
    np.save(os.path.join(save_path, "recurrent.npy"), recurrent)
    np.save(os.path.join(save_path, "start.npy"), start)
    np.save(os.path.join(save_path, "decoder.npy"), decoder)
    print("Веса обновлены")


def load_model(save_path):
    """Функция для загрузки весов модели"""
    embed = np.load(os.path.join(save_path, "embed.npy"))
    recurrent = np.load(os.path.join(save_path, "recurrent.npy"))
    start = np.load(os.path.join(save_path, "start.npy"))
    decoder = np.load(os.path.join(save_path, "decoder.npy"))
    return embed, recurrent, start, decoder
