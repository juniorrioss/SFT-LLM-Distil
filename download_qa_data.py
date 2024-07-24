
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()
dataset = load_dataset("corejur/aviacao_qa_20_v3", token=os.environ['HF_KEY'])


def gen_exemplo(df, file_name, split, questions_per_prompt=23):

    subset = df.query(f"file_name == '{file_name}'")
    document = subset['document'].iloc[0]
    questions = subset['prompt']
    all_answers = subset['answer']
    test_answers = []
    texts = []

    for i in range(0, len(questions), questions_per_prompt):
        questions_this_iteration = questions[i:i+questions_per_prompt]
        answers_this_iteration = all_answers[i:i+questions_per_prompt]

        questions_this_iteration = str({f"Pergunta{i+1}": qst for i, qst in enumerate(questions_this_iteration)})
        answers_this_iteration = str({f"Resposta{i+1}": ans for i, ans in enumerate(answers_this_iteration)})


        text = f"""Você é um excelente assistente de um escritorio juridico. Dado um contexto e diversas pergunta, você deve procurar no contexto a resposta para a pergunta. RETORNE SOMENTE COMO PEDIDO NA QUESTÃO, utilizando o contexto como única fonte de informação. Responda apenas com Sim ou Não. Sua resposta deve ser no formato JSON, com as chaves "Resposta1" contendo a resposta da "Pergunta1" e sucessivamente."""

        text += '\n ###Contexto: \n' + document

        text += '\n ###Perguntas: \n' + questions_this_iteration

        text += "\n ###Respostas: \n"

        if split == "train":
            text +=  answers_this_iteration
        else:
            test_answers.append(answers_this_iteration)

        text = text.replace('\n\n', '\n').replace('\t',' ')

        texts.append(text)


    row = {'text': texts, 'filename': file_name}
    if split == "test":
        row['answer'] = test_answers
    return row


df_splits = {}
df_splits['train'] = dataset['train'].to_pandas()
df_splits['test'] = dataset['test'].to_pandas()


os.makedirs('./data/qa_aviacao', exist_ok=True)
for split in df_splits.keys():
    df_split = df_splits[split]
    df_split = df_split.sort_values('file_name')

    df_split = pd.DataFrame.from_dict([gen_exemplo(df_split, file_name, split) for file_name in df_split['file_name'].unique()])
    # text e answer uma lista com todos os prompts/respostas para cada doc, com o explode a gente deixa cada um em uma coluna diferente
    if split == "train":
        df_split = df_split.explode(['text']).reset_index(drop=True)
    else:
        df_split = df_split.explode(['text', 'answer']).reset_index(drop=True)

    save_path = f"./data/qa_aviacao/aviacao_23q_{split}.json"
    df_split.to_json(save_path)
    print(f"***SALVO SPLIT {split.upper()} EM {save_path}***\n\n")
    
    
    # df_split.to_csv(save_path, index=None, escapechar='\\', mode='a')