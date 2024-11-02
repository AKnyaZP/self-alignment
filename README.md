# Self-Alignment with LLM for Russian Instructions

## Описание :page_with_curl:
Этот проект использует датасет  <a href='https://huggingface.co/datasets/OpenAssistant/oasst1'>OpenAssistant/oasst1</a> для обучения модели LLM, которая генерирует ответы на русскоязычные инструкции. Основная цель проекта — исследовать подходы к выравниванию модели (self-alignment) для достижения определенного стиля ответов. Я буду использовать *google/gemma-2-2b-it* в качестве языковой модели, для которой будет проводиться self-alignment.    
**WandB report** :arrow_down:  
 <a href="https://wandb.ai/ai-sigma/alignment-gemma?nw=nwuserknyazev_artem">
   <img src="https://avatars.githubusercontent.com/u/26401354?s=200&v=4#50x50" alt="Weights & Biases Logo" width="130" height="130">
 </a>


## Структура проекта :file_folder:
- `data/`: Директория для хранения данных (не включены в репозиторий). Датасеты с инструкциями на русском языке.
    - `data/data4alignment`: В этой директории хранятся данные именно для выравнивания модели. 
- `notebooks/`:  `/eda.ipynb`- ноутбук с **EDA** и **визуализацией данных**, а также `/data4alignment.ipynb` - ноутбук с созданием **датасета для элаймента**, посредством *генерации ответов* на инструкции и их *оценкой* по критериям из таблицы (в качестве *label* я использовал *усреднённую оценку* ответа по всем заданным критериям).
- `src/`: Здесь хранится код для запуска дообученной модели
- `experiments/`: эксперименты и результаты
- `models/`: здесь хранится дообученная модель
    - `models/fine_tuned_gemma_model_dpo/`: здесь хранятся все зависимости для запуска модели. Пример того, как можно её запустить находится в `src/launched_model.py`
    - `models/results/`: Здесь хранятся файл с информацией об обучении модели и зависимости для запуска модели вместе *peft*. Файлы `./optimizer.pt`, `./scheduler.pt`, и `./trainer_state.json` содержат информацию о состоянии обучения. Они полезны в случае возобновления обучения с того же места или проведения дополнительного дообучения.


## Какой стиль ответов будет достигаться посредством самовыравнивания? :pencil2:
*Достигаться будет стиль кратких и содержательных ответов*

### Почему краткие и содержательные ответы?
#### Улучшение качества ответов:
* Короткие и полезные ответы чаще всего содержат наиболее важную и релевантную информацию.
* Их легче читать и понимать, что особенно важно для широкого круга пользователей.
#### Повышение эффективности:
* Более краткие ответы дают пользователю возможность быстрее получить нужную информацию.
* Это особенно важно в контексте диалоговых систем, где время ответа критично.
#### Улучшение пользовательского опыта:
* Короткие, но полезные ответы делают систему более дружелюбной и удобной для использования.
* Пользователи меньше вероятны будут чувствовать себя перегруженными избыточными данными.


## Шкала критериев :clipboard:
### По этой шкале модель будет оценивать саму себя 

| Критерий      | Описание                                                                                                                                  | Оценка (по 5-балльной шкале) |
|---------------|--------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| **Краткость** | Ответ содержит только необходимую информацию, без лишних деталей. <br> 5 — идеально краткий; 1 — перегружен ненужной информацией.           | 1-5                           |
| **Полезность**| Ответ полностью решает задачу или отвечает на вопрос. <br> 5 — полностью полезен; 1 — не имеет отношения к вопросу.                         | 1-5                           |
| **Точность**  | Ответ достоверен и корректен. <br> 5 — полностью точен; 1 — содержит ошибки.                                                               | 1-5                           |
| **Ясность**   | Ответ понятен и не содержит сложных формулировок или терминов, которые могут сбить с толку. <br> 5 — легко понять; 1 — запутан или неясен. | 1-5                           |
| **Лаконичность**| Ответ экономен в словах, избегает повторений и избыточности. <br> 5 — максимально лаконичен; 1 — многословен и избыточен.                  | 1-5                           |  

Для self-alignment модели мы используем инструкции на русском языке из датасета <a href='https://huggingface.co/datasets/OpenAssistant/oasst1'>OpenAssistant/oasst1</a>, а также ответы самой модели и её оценка по критериям из таблицы выше.  
Для генерации ответов и одновременной оценки этих ответов по шкале критериев, я использовал промпт, который модель получала на вход вместе с инструкцией:  

**PROMPT:**  
`("Отвечай на русском языке. " + instruction + ". Дай краткий и содержательный ответ, затем оцени ответ от 1 до 5 по критериям: " +  
criteria + ". Важно! Ответ должен быть в таком формате: ответ, оценка от 1 до 5 без лишнего текста и символов.  
Не нужно вставлять объяснения ответа и его качества.")` 

Рассмотрим этот промпт подробнее:
* `Отвечай на русском языке`: Эта часть важна в нашем запросе, так как *Gemma* может начать отвечать на английском, если инструкция содержит английские слова.
* `instruction`: Это входная инструкция, на которую должна отвечать наша языковая модель
* `Дай краткий и содержательный ответ, затем оцени ответ от 1 до 5 по критериям`: Здесь мы просим модель дать именно краткий и содержательный ответ, для его дальнейшей оценки
* `criteria`: Здесь хранятся все критерии, по которым нужно оценивать ответ модели
* `Важно! Ответ должен быть в таком формате: ответ, оценка от 1 до 5 без лишнего текста и символов. Не нужно вставлять объяснения ответа и его качества`  
  **или**
* `Важно! Ответ должен быть в формате JSON без лишнего текста: {\"answer\": \"[твой ответ]\", \"estimate\": [оценка от 1 до 5]}.`: эта часть промпта одна из самых главных. Здесь мы просим генерировать ответ в определённой форме для более просто извлечения самого ответа и его оценки.

## Alignment Technique :pencil:
Для выравнивания модели я использовал одну из самых распространённых техник - **DPO (Deep Prompt Optimization)**
Для выравнивания модель с помощью этой техники требуется датасет `| Промпт | Ответ | Оценка (по 5-балльной шкале) |`  
* `Промпты` или `Инструкции` я взял из датасета <a href='https://huggingface.co/datasets/OpenAssistant/oasst1'>OpenAssistant/oasst1</a>  и выделил из них только инструкции на русском языке. Как я это сделал, можно посмотреть в здесь `notebooks/eda.ipynb`
* `Ответы` генерировались самой моделью *google/gemma-2-2b-it* на каждую инструкцию
* `Оценка` - Обычно для проведения *alignment* с помощью **DPO** используется **RLHF(Reinforcement Learning from Human Feedback)** - метод, который сочетает в себе принципы подкрепленного обучения и обратную связь человека для улучшения поведения искусственного интеллекта или оценка формируется посредством использования другой более сильной языковой модели. Но так как мы проводим *self-alignment*, то и модель должна оценивать свои ответы сама. 

## Оценка ответа выравненной модели (Человеком) :sunglasses::speech_balloon:
**PROMPT:** `Отвечай на русском языке. Привет Gemma! Расскажи мне как провести self-alignment модели *google/gemma-2-2b-it*?`

**ANSWER:**  
```
**Self-Alignment:**
* Это метод, который позволяет обучить модель к задаче без использования больших данных.
* Он использует информацию из самого текста для обучения и улучшения качества ответов.
  
**Google/Gemma:**
* Google разработал модель искусственного интеллекта с открытым исходным кодом, которая называется Gemma.
* Она доступна для всех желающих использовать и модифицировать.

**Как проводить Self-аlignment моделей Google/Gemma:**
1. **Подготовка данных:** Вам нужно собрать набор данных, содержащий примеры текстов и соответствующих ответов.  Это может быть текст из книг, статей или других источников.
    * Важно: данные должны быть качественными и релевантными для задачи, которую вы хотите решить.   
2. **Обучение модели:** После того, как вы собрали данные, вам нужно обучить модель Gemma. Для этого можно воспользоваться инструментами, которые предоставляются в рамках проекта Gemma.
3. **Оценка результатов:** После обучения модели необходимо оценить ее результаты. Вы можете сделать это, используя метрики, такие как точность, полноту и соответствие.
4. **Адаптация:** Если результаты не удовлетворительны, вы можете адаптировать модель путем изменения параметров обучения или добавления новых данных.

**Важно помнить:** Self-аlignment - это сложный процесс, требующий времени и усилий. Но если вы готовы потратить время и усилия, вы сможете создать более эффективные модели AI.
```
**Учитывая, что единственный системный промпт - это "отвечай на русском языке" (я не просил модель отвечать кратко и содержательно), ответ модель очень даже неплохой.  
Единственная часть в ответе, которая может показаться лишней - это:**  
```
Google/Gemma:  
Google разработал модель искусственного интеллекта с открытым исходным кодом, которая называется Gemma.  
Она доступна для всех желающих использовать и модифицировать.
```  
**По моему мнению, Gemma научилась давать краткие и содержательные ответы, согласно шкале критериев, указанных выше.** 

## Установка и запуск выравненной модели *google/gemma-2-2b-it* :computer:
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/AKnyaZP/self-alignment.git
   cd self-alignment
   
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt

3. Перейдите в директорию `src`:
   ```bash
   cd src
   
4. Запустите файл `launch_model.py`:
   ```bash
   python launch_model.py

5. У вас появится сообщение `Очистить кэш и вызвать сборщик мусора? Y/n` (Код может очистить кэш, если вам это нужно, потому что не у всех устройств много памяти):
   ```bash
   Очистить кэш и вызвать сборщик мусора? Y/n: 

6. Введите свой запрос в ответ на сообщение `Введите ваш запрос: `:
   ```bash
   Введите ваш запрос: <your querry>

7. Далее вставьте свой `huggingface_token`, в ответ на сообщение `Введите ваш токен Hugging Face: ` и нажмите `Enter`.
   Создать huggingface токен можно тут :arrow_right: <a href="https://huggingface.co/settings/tokens"> <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF_token" width="30" height="30"></a>
   ```bash
   Введите ваш токен Hugging Face: <your_token>

8. Ожидайте ответа модели)) 
