# Port of llms-for-cta/cta.py (commit e5845e8) to our vLLM (GLM-4-9B). Prompts, examples, retrieval and metrics are the original code;
# changed: the chat client, temperature 0 / seed 42, reply read from `reasoning` when `content` is empty, the final [...] list of a multi-column answer is parsed,
# and the main block runs ONE experiment per call.
import json
import pandas as pd
import numpy as np
import ast
import re
from openai import OpenAI
from numpy.linalg import norm
from sklearn.metrics import f1_score, accuracy_score, classification_report


# ---- RAG4CTA port: every chat call of the original code goes to our vLLM (GLM-4-9B) instead of OpenAI ----------------------------
import os, json as _json
_vllm = OpenAI(base_url=os.environ["VLLM_BASE_URL"], api_key=os.environ["VLLM_API_KEY"], timeout=300)
_MAXTOK = int(os.environ.get("CTA_MAX_TOKENS", "1000"))
_USAGE = {"prompt": 0, "completion": 0, "calls": 0}

class _Msg:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content): self.message = _Msg(content)
class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]

class _Completions:
    def create(self, model=None, messages=None, **kw):
        for attempt in range(3):
            try:
                r = _vllm.chat.completions.create(model=os.environ["VLLM_MODEL"], messages=messages, temperature=0, seed=42, max_tokens=_MAXTOK)
                break
            except Exception as e:
                if attempt == 2: raise
        m = r.choices[0].message
        text = m.content or getattr(m, "reasoning", None) or ""    # this deployment sometimes puts the reply in `reasoning`
        if r.usage:
            _USAGE["prompt"] += r.usage.prompt_tokens; _USAGE["completion"] += r.usage.completion_tokens
        _USAGE["calls"] += 1
        return _Resp(text)

class _Chat:
    completions = _Completions()

class _Client:
    chat = _Chat()

client = _Client()

import urllib.request, time as _time

def mistral_embed(texts, model="mistral-embed"):
    """RAG4CTA port: embeddings from the Mistral API (OpenAI has no credit, our vLLM server has no embedding route).
    The reference vectors shipped with the repo are OpenAI ada-002 vectors, so they are recomputed with this model (embed_refs.py)."""
    texts = [t.replace("\n", " ")[:6000] or " " for t in texts]
    for attempt in range(8):
        try:
            req = urllib.request.Request("https://api.mistral.ai/v1/embeddings",
                data=json.dumps({"model": model, "input": texts}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["MISTRAL_API_KEY"]})
            with urllib.request.urlopen(req, timeout=120) as r:
                return [d["embedding"] for d in json.load(r)["data"]]
        except Exception:
            if attempt == 7: raise
            _time.sleep(3 * (attempt + 1))

def get_embedding(text, model="mistral-embed"):
   return mistral_embed([text])[0]

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    norm_a = norm(vector_a)
    norm_b = norm(vector_b)

    similarity = dot_product / (norm_a * norm_b)
    return similarity


def single_column_prompting(dataset, few_shot=True):
    results = []
    #cnt = 0

    # Few-shot examples dictionary
    single_column_examples = {
    "example1": ("Column 1: Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware District of Columbia Florida Georgia Hawaii Idaho", "state"),
    "example2": ("Column 1: DJ REVOLUTION DJ REVOLUTION KANYE WEST ATMOSPHERE JAY-Z", "artist"),
    "example3": ("Column 1: South Orange Orlando Jupiter San Francisco San Francisco San Francisco San Francisco New York", "city"),
    "example4": ("Column 1: Alabama Florida State Ohio State Baylor Stanford Oregon Clemson Auburn Texas A&M Oklahoma St. Missouri South Carolina", "team"),
    }

    for i in range(len(dataset)):
        list_temp = dataset['data'][i][0:250]

        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma:"
                    "sex, category, album, status, origin, format, day, location, notes, duration, nationality, region, club, address, rank, name, "
                    "position, description, country, state, city, code, symbol, isbn, age, type, gender, team, year, company, result, artist."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Your instructions are: 1. Look at the column and the types given to you. 2. Examine the values of the column. "
                    "3. Select a type that best represents the meaning of the column. 4. Answer with the selected type only, and print the type only once. "
                    "The format of the answer should be like this: type  Print 'I don't know' if you are not able to find the semantic type."
                ),
            },
        ]

        # Conditionally add few-shot examples
        if few_shot:
            for _, (example_input, example_output) in single_column_examples.items():
                messages.append({"role": "user", "content": f"Classify this column:\n\n{example_input}"})
                messages.append({"role": "assistant", "content": example_output})

        # Add actual input
        messages.append({"role": "user", "content": f"Classify this column:\n\n{list_temp}"})

        response = client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="gpt-4.1-mini",
            messages=messages,
        )
        results.append(response.choices[0].message.content.strip())

        #cnt += 1
        #if cnt % 10 == 0:
        #    print(cnt)

    return results



def RAG_prompting(test_set, train_set):
    from ast import literal_eval

    correct_count = 0
    results = []

    for i in range(len(test_set)):
        query_embedding = get_embedding(test_set['data'][i])

        # Initialize with low scores and placeholders
        few_shot_tuples = [(-float('inf'), "class", "data")] * 4

        for j in range(len(train_set)):
            candidate_embedding = literal_eval(train_set["embedding"][j])
            similarity = cosine_similarity(query_embedding, candidate_embedding)

            # Find the lowest scoring example to potentially replace
            min_score = min(few_shot_tuples, key=lambda x: x[0])
            min_index = few_shot_tuples.index(min_score)

            if similarity > min_score[0]:
                few_shot_tuples[min_index] = (similarity, train_set["class"][j], train_set["data"][j])

        # Prepare prompt parts
        examples = [
            {"user": f"Column 1: {t[2][:500]}", "assistant": t[1]}
            for t in sorted(few_shot_tuples, key=lambda x: -x[0])
        ]
        user_input = {"role": "user", "content": f"Column 1: {test_set['data'][i][:500]}"}

        # Quick correctness check
        gt_class = test_set["class"][i]
        if any(gt_class == example["assistant"] for example in examples):
            correct_count += 1

        # Build message sequence
        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to classify the columns of a given table with only one of the following classes "
                    "that are separated with comma: sex, category, album, status, origin, format, day, location, notes, "
                    "duration, nationality, region, club, address, rank, name, position, description, country, state, city, "
                    "code, symbol, isbn, age, type, gender, team, year, company, result, artist."
                ),
            },
            {
                "role": "system",
                "content": (
                    "Your instructions are: 1. Look at the column and the types given to you. "
                    "2. Examine the values of the column. 3. Select a type that best represents the meaning of the column. "
                    "4. Answer with the selected type only, and print the type only once. "
                    "The format of the answer should be like this: type  Print 'I don't know' if you are not able to find the semantic type."
                ),
            },
        ]

        for ex in examples:
            messages.append({"role": "user", "content": ex["user"]})
            messages.append({"role": "assistant", "content": ex["assistant"]})
        messages.append(user_input)

        # Query GPT
        response = client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="gpt-4.1-mini",
            messages=messages,
        )
        results.append(response.choices[0].message.content.strip())

    return results




# settings: single_shot, few_shot, COT, RAG
def multi_column_prompt_gpt(dataset, setting="single_shot"):
    pred_answer = []
    gt_answer = []
    table_ids = dataset['table_id'].unique()

    # Few-shot examples for multi-column classification
    multi_column_examples = {
    "example1": ("""
    Classify these column: 

    Column 1: Away Away Away Away Away Away Away Home
    Column 2: Final Final Final Final Final Final Final Final
    Column 3: Loss Loss Loss Win Win Loss Win Win
    """, "['location', 'status', 'result']"),

    "example2": ("""
    Classify these column: 

    Column 1: Jeff Fink nathaniel wells Brook Bielen Ken Hahn Andrew Jansy michael thoen Andrew Hayes Nick Graves Michael Jones Mark Deresta Lloyd Connelly Andy Crump Ricardo Medina Justin Huggins Erik Denninghoff Marcelo Heredia Adam Keir Dante Solano Reynaldo Ortiz rudy vega Mark Vickers
    Column 2: Dimwits Rogue / Mt. View Cycles Wolverine Mountain View Cycles/Subway DNR Cycling Rogue / Mt. View Cycles DNR Cycling Camas Bike and Sport DNR Cycling Upper Echelon Fitness and Rehabilitation
    Column 3: White Salmon Mosier Portland Portland Portland White Salmon Hood River Portland Hood River Portland Oregon City Vancouver Portland Portland Vancouver Portland Portland Portland Forest Grove
    """, "['name', 'team', 'city']"),

    "example3": ("""
    Classify these column: 

    Column 1: Greenhouse Gases Greenhouse Gases Greenhouse Gases Greenhouse Gases Greenhouse Gases
    Column 2: C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide (d13C (CO2)) C13/C12 in Carbon Dioxide
    Column 3: Flask Flask Flask Flask Flask
    """, "['category', 'name','type']"),

    "example4": ("""
    Classify these column: 

    column 1: Introduction to Paddle Boarding Intro to Paddle Boar Intro to Paddle Boar
    column 2: Phillips Lake Park Phillips Lake Park Phillips Lake Park
    column 3: Unavailable Unavailable Unavailable
    """, "['description', 'location', 'status']")
    }

    multi_column_examples_cot = {
    "example1": ("""
    Classify these column: 

    Column 1: TV TV Other TV TV Web
    Column 2: 2006 2009 2012 2010 2001 2003 
    Column 3: Want to Watch Want to Watch Want to Watch Want to Watch Want to Watch Want to Watch
    """, """
    Reasoning:
    Looking at the table row by row like the first row is "TV," "2006," "Want to Watch," it seems that the first column refers to the type of media, the second column refers to the year associated with each entry, and the third column indicates the status or interest in watching. So, the types would be ["type", "year", "status"].
    """),

    "example2": ("""
    Classify these column: 

    Column 1: Wednesday Thursday Friday Saturday Sunday Monday
    Column 2: Home Home Away Away Home Home 
    Column 3: Final Final Final Final Final Final
    Column 4: Loss Win Loss Loss Loss Win
    """, """
    Reasoning:
    Looking at the table row by row like the first row is "Wednesday," "Home," "Final," "Loss," it seems that the first column refers to the day of the event, the second column refers to the location, the third column indicates the status of the event, and the fourth column shows the result of the event. So, the types would be ["day", "location", "status", "result"]
    """),

    "example3": ("""
    Classify these column: 

    Column 1: 24 24 24 24 24 24
    Column 2: Los Angeles Los Angeles Los Angeles Los Angeles Los Angeles Los Angeles
    Column 3: Hungary Hungary Hungary Hungary Hungary Hungary
    Column 4: 19 4 19 9 5 5
    """, """
    Reasoning:
    Looking at the table row by row like the first row is "24," "Los Angeles," "Hungary," "19," it seems that the first column refers to age of a person based on its columns's range, the second column refers to a city, the third column refers to a country, and the fourth column could represent a rank or some form of numeric code. So, the types would be ["code", "city", "country", "rank"]
    """),

    "example4": ("""
    Classify these column: 

    Column 1: Athina Athina Athina Athina Athina Athina Athina Athina Athina
    Column 2: Australia Australia Australia Australia Australia Australia Australia Australia Australia
    Column 3: 2 1 2 2 1 1 1 1 2
    Column 4: CUB 6, AUS 2 AUS 1, JPN 0 CUB 4, AUS 1 TPE 3, AUS 0 AUS 6, ITA 0 AUS 9, JPN 4 AUS 11, GRE 6 AUS 22, NED 2 CAN 11, AUS 0
    """, """
    Reasoning:
    Looking at the table row by row like the first row is "Athina," "Australia," "2," "CUB 6, AUS 2," it seems that the first column refers to the city where the event took place, the second column refers to a team, the third column indicates the rank of the team based on some criteria, and the fourth column shows the result of a game or match. So, the types would be ["city", "team", "rank", "result"]
    """)
    }

    for cnt, table_id in enumerate(table_ids):
        # Aggregate column data for a given table
        filtered_rows = dataset[dataset['table_id'] == table_id]['data'].iloc[:200]    ## take the first 200 samples if more than 200 samples
        list_temp = "\n".join([f"column {i + 1}: {val}" for i, val in enumerate(filtered_rows)])

        gt_answer.append(str(list(dataset[dataset['table_id'] == table_id]['class'].values)))  

        #print(gt_answer)
        # Build the base system prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma:"
                    " sex, category, album, status, origin, format, day, location, notes, duration, nationality, region, club, address, rank, name, "
                    "position, description, country, state, city, code, symbol, isbn, age, type, gender, team, year, company, result, artist."
                )
            },
            {
                "role": "system",
                "content": (
                    "Your instructions are: 1. Look at the columns and the types given to you. 2. Examine the values of the columns. "
                    "3. Select a type that best represents the meaning of each column. 4. Answer with the selected type only. "
                    "The format of the answer should be like this: ['type1', 'type2', 'type3']. Print 'I don't know' if you are not able to find the semantic type."
                )
            },
        ]

        # Add few-shot examples if enabled
        if setting == "few_shot":
            for example, answer in multi_column_examples.values():
                messages.append({"role": "user", "content": example})
                messages.append({"role": "assistant", "content": answer})
        elif setting == "COT":
            for example, answer in multi_column_examples_cot.values():
                messages.append({"role": "user", "content": example})
                messages.append({"role": "assistant", "content": answer})

        # Add the current table to be classified
        messages.append({"role": "user", "content": list_temp})

        # Query GPT
        response = client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="gpt-4.1-mini",
            messages=messages
        )

        pred_answer.append(response.choices[0].message.content.strip())

        #if cnt % 10 == 0:
        #    print(cnt)

    return pred_answer, gt_answer

# function for making each tables a serialized string and separating columns with | to implement RAG on the test tables
def format_table(group):
    columns = []
    for idx, sub_group in group.groupby('col_idx'):
        col_string = f"Column {idx}: " + " ".join(sub_group['data'][0:200].tolist())
        columns.append(col_string)
    return " | ".join(columns)


# creates the correct formatting of few shot examples with the desired output
def example_creator(group):
    data_string = "Classify these columns:\n\n"
    class_list = []

    for idx in sorted(group['col_idx'].unique()):
        column_data = group[group['col_idx'] == idx]['data'].values[0]
        column_class = group[group['col_idx'] == idx]['class'].values[0]
        data_string += f"Column {idx}: {column_data}\n"
        class_list.append(column_class)

    class_string = f'\n{class_list}'

    return data_string, class_string


def RAG_prompting_multi_column(test_set, training_set_with_embedding, original_training_set):
    grouped = original_training_set.groupby('table_id')
    results = grouped.apply(example_creator)
    grouped_test = test_set.groupby('table_id')
    results_test = grouped_test.apply(example_creator)

    pred_answer = []
    gt_answer = []
    checked_ids = []

    for i in range(len(test_set)):
        if test_set["table_id"][i] in checked_ids:
            continue
        else:

            # Step 1: Create the formatted string for the current table in the test set
            current_table = test_set[test_set['table_id'] == test_set['table_id'].iloc[i]]
            formatted_test_string = format_table(current_table)

            # Step 2: Get the embedding for the formatted string
            list1 = get_embedding(formatted_test_string)  # Ensure it's a 1D array

            # Step 3: Initialize few-shot tuples with the lowest cosine similarities
            few_shot_tuples = [(-float('inf'), "class", "data", "table_id") for _ in range(4)]
            min_index = None

            # Step 4: Find the most similar strings in the train set
            for j in range(len(training_set_with_embedding)):
                list2 = ast.literal_eval(training_set_with_embedding["embedding"][j])
                cos_sim = cosine_similarity(list1, list2)  # Cosine similarity between two 1x1536 arrays

                min_value = min(few_shot_tuples, key=lambda x: x[0])[0]

                if cos_sim > min_value:
                    min_index = few_shot_tuples.index(min(few_shot_tuples, key=lambda x: x[0]))
                    data_string, class_string = results[training_set_with_embedding["table_id"][j]]
                    few_shot_tuples[min_index] = (cos_sim, class_string, data_string, training_set_with_embedding["table_id"][j])

            # Step 5: Prepare examples from the original DataFrame
            examples = []
            answers = []
            for index, (_, class_label, _, table_id) in enumerate(few_shot_tuples):
                similar_table = original_training_set[original_training_set['table_id'] == table_id]
                example = f"Classify these column:\n"
                for col_idx in similar_table['col_idx'].unique():
                    col_data = " ".join(similar_table[similar_table['col_idx'] == col_idx]['data'].tolist())
                    example += f"Column {col_idx}: {col_data}\n"
                answer = f'["{class_label}"]'
                examples.append(f'example{index + 1}_multi_reason = """\n{example}"""')
                answers.append(f'answer{index + 1}_multi_reason = """\n{answer}"""')
            data_string_test, class_string_test = results_test[test_set["table_id"][i]]

            # Combine examples and answers into the LLM prompt
            response = client.chat.completions.create(
            #model="gpt-3.5-turbo",
            model="gpt-4.1-mini",
            messages=[
                    {"role": "system","content": "Your task is to classify the columns of a given table with only one of the following classes that are seperated with comma: sex, category, album, status, origin, format, day, location, notes, duration, nationality, region, club, address, rank, name, position, description, country, state, city, code, symbol, isbn, age, type, gender, team, year, company, result, artist."},
                    {"role": "system","content": "Your instructions are: 1. Look at the columns and the types given to you. 2. Examine the values of the columns. 3. Select a type that best represents the meaning of each column. 4. Answer with the selected type only. the format of the answer should be like this: ['type1', 'type2', 'type3']   Print 'I don't know' if you are not able to find the semantic type."},
                    {"role": "user", "content": few_shot_tuples[0][2]},
                    {"role": "assistant", "content": few_shot_tuples[0][1]},
                    {"role": "user", "content": few_shot_tuples[1][2]},
                    {"role": "assistant", "content": few_shot_tuples[1][1]},
                    {"role": "user", "content": few_shot_tuples[2][2]},
                    {"role": "assistant", "content": few_shot_tuples[2][1]},
                    {"role": "user", "content": few_shot_tuples[3][2]},
                    {"role": "assistant", "content": few_shot_tuples[3][1]},
                    {"role": "user", "content": data_string_test}
                ]
            )
            res=response.choices[0].message.content.strip()
            pred_answer.append(res)
            gt_answer.append(class_string_test.strip())

            #if set(class_string_test) == set(few_shot_tuples[0][1]) or set(class_string_test) == set(
            #        few_shot_tuples[1][1]) or set(class_string_test) == set(few_shot_tuples[2][1]) or set(
            #        class_string_test) == set(few_shot_tuples[3][1]):

            checked_ids.append(test_set["table_id"][i])

            #if i % 10 == 0:
            #    print(f"Processed {i} tables")

    return pred_answer, gt_answer

def safe_literal_eval(entry):
    lists = re.findall(r"\[[^\[\]]*\]", entry)      # RAG4CTA port: use the final list of the answer when there is text around it
    if lists:
        entry = lists[-1]
    try:
        val = ast.literal_eval(entry)
        if isinstance(val, list):
            return val
        else:
            return [str(val)]  # wrap string like "I dont know" into a list
    except (ValueError, SyntaxError):
        return [entry]  # fallback if completely unparseable

def print_eval_result (gt, pred):

    # Compute Accuracy
    accuracy = accuracy_score(gt, pred)
    # Compute Micro F1 Score
    micro_f1 = f1_score(gt, pred, average='micro')
    macro_f1 = f1_score(gt, pred, average='macro')

    # Compute Per-class F1 Score
    class_report = classification_report(gt, pred)

    print(f"Accuracy: {accuracy},\t Micro F1 Score: {micro_f1},\t Macro F1 Score: {macro_f1}")
    print("Per-class F1 Score:")
    print(class_report)

def print_single_col_res(gt_flat, pred_flat, exp):

    print(f'LOG for {exp}')
    print(f'predicted labels: {pred_flat}')
    print(f'ground truth labels: {gt_flat}')
    print_eval_result(gt_flat, pred_flat)
    print("\n")

def print_multi_col_res(gt, pred, exp):

    print(f'LOG for {exp}')
    print(f'predicted labels: {pred}')
    print(f'ground truth labels: {gt}')
    pred = [re.sub(r"n't", "nt", entry) for entry in pred]  # quotation in fields is a problem for the next func
    pred_flat = [label for entry in pred for label in safe_literal_eval(entry)]
    gt_flat = [label for entry in gt for label in ast.literal_eval(entry)]

    # print the labels (predicted and ground truth) in flat form for logging
    print(f'predicted labels (flat): {pred_flat}')
    print(f'ground truth labels (flat): {gt_flat}')
    print_eval_result(gt_flat, pred_flat)
    print("\n")
    
if __name__ == "__main__":
    import sys, time
    exp = sys.argv[1]                      # sc_zero sc_few sc_rag mc_zero mc_few mc_cot mc_rag
    test_file = sys.argv[2] if len(sys.argv) > 2 else "data/test_set.csv"
    out = f"outputs/{exp}"
    os.makedirs("outputs", exist_ok=True)
    dataset = pd.read_csv(test_file)
    t0 = time.time()
    if exp.startswith("sc_"):
        gt_flat = list(dataset['class'].values)
        if exp == "sc_zero": pred_flat = single_column_prompting(dataset, False)
        elif exp == "sc_few": pred_flat = single_column_prompting(dataset, True)
        else: pred_flat = RAG_prompting(dataset, pd.read_csv("data/RAG_2000_sample_mistral.csv"))
        print_single_col_res(gt_flat, pred_flat, exp)
        json.dump({"gt": gt_flat, "pred": pred_flat, "usage": _USAGE, "seconds": time.time() - t0}, open(out + ".json", "w"))
    else:
        if exp == "mc_zero": pred, gt = multi_column_prompt_gpt(dataset, "single_shot")
        elif exp == "mc_few": pred, gt = multi_column_prompt_gpt(dataset, "few_shot")
        elif exp == "mc_cot": pred, gt = multi_column_prompt_gpt(dataset, "COT")
        else:
            pred, gt = RAG_prompting_multi_column(dataset, pd.read_csv("data/trainset_for_multi_rag_embeds_mistral.csv"), pd.read_csv("data/trainset_for_multi_rag.csv"))
        print_multi_col_res(gt, pred, exp)
        json.dump({"gt": gt, "pred": pred, "usage": _USAGE, "seconds": time.time() - t0}, open(out + ".json", "w"))
