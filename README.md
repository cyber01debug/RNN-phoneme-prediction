# RNN (рекуррентная нейронная сеть) для предсказания фонемы
Простая рекуррентная нейронная сеть (RNN), обученная предсказывать следующую фонему в слове.
# Рекуррентная Нейронная сеть для предсказания следующих фонем в слове
Этот репозиторий содержит реализацию простой рекуррентной нейронной сети (RNN), обученной для предсказания следующей фонемы в последовательности. Код написан на Python с использованием библиотеки NumPy для математических операций.
Код загружает набор данных с фонетической транскрипцией слов, строит словарь фонем, и обучает RNN предсказывать следующую фонему в последовательности.

## Основные шаги работы алгоритма:
1.  **Подготовка данных:**
    - загрузка данных из файла 'phon.txt'
    - токенизация слов в фонемы
    - создание словаря фонем и их уникальных индексов
2.  **Модель:**
    - инициализация матриц для эмбеддингов, рекуррентных связей и декодера
    - использование прямого распространения для вычисления предсказаний и ошибки
    - использование обратного распространения для обновления весов модели
3.  **Обучение:**
    - обучение модели в течение некотрого количества итераций (в нашем случае 250т. и 500т.)
    - вывод номера итерации и значения перплексии для оценки модели
4.  **Тестирование:**
    - использование обученной модели для предсказания следующей фонемы в слове
    - сравнение предсказания с фонемой, которая должна быть

# Использование 
1. Клонирование репозитория
Клонируйте репозиторий в пустую папку
```bash 
 git clone https://github.com/cyber01debug/RNN-phoneme-prediction.git
```
2. Установка необходимых библиотек
В нашем случае устанавливается только библиотека NumPy остальные библиотеки уже встроены в Python
```bash
pip install -r requirements.txt
```
3. Подготовить данные:
  - cоздайте файл 'phon.txt' в корне репозитория или можете уже созданный датасет
  - в каждой строке файла должно быть слово, записанное фонетической транскрипцией, с фонемами, разделенными пробелами
4. Запуск
Запустите код через командную строку папки с проектом
```bash
python RNN_next_phoneme.py
```
## Файлы 
В репозитории уже присутсвует обязательный файл с датасетом (phon.txt). Файл имеет формат:
```bash
j u r' i
t r' i f a n a f
a b' m' e n
```
Также в репозитории присутствует не только файл формата .py, но и .ipynb для более удобного отслеживания этапов кода.
