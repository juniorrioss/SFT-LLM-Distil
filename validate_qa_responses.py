import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, classification_report



test_file = f"data/qa_aviacao/aviacao_23q_test.json"
data_path = f"outputs/qwen7b-QA_r8.json"
df = pd.read_json(data_path)
#import pdb;pdb.set_trace()

# questions = []
# for text in df['text']:
#     questions.extend(list(eval(text.split("###Perguntas:\n")[1].split("\n###Respostas:")[0]).values()))


answers, outputs = [], []

for i, row in df.iterrows():
  try:
    ans_json = eval(row['answer'])
    out_json = eval(row['output'])
    answers.append(ans_json)
    outputs.append(out_json)
  except:
    print(i)
    print(row)

# df['answer'] = df['answer'].apply(eval)
# df['output'] = df['output'].apply(eval)

# total_questions_truncated = 0
# total_questions_add = 0
# for row_real, row_out in zip(df['answer'], df['output']):
#     row_real = list(row_real.values())
#     row_out = list(row_out.values())

#     total_questions_truncated += max(0, len(row_out) - len(row_real))
#     total_questions_add += max(0, len(row_real) - len(row_out))

#     row_out = row_out[:len(row_real)]
#     row_out.extend(['Nao'] * (len(row_real) - len(row_out)))
#     row_out = ['sim' if answer.lower()[0] == 's' else 'nao' for answer in row_out]
#     row_real = ['sim' if answer.lower()[0] == 's' else 'nao' for answer in row_real]
#     answers.extend(row_real)
#     outputs.extend(row_out)

# print("*****"*10)
# print(f"- TOTAL QUESTOES NO DATASET: {len(questions)}")
# print(f"- TOTAL DE QUESTOES NAO RESPONDIDAS: {total_questions_add}")
# print(f"- TOTAL QUESTOES RESPONDIDAS A MAIS: {total_questions_truncated}")
# print("*****"*10)

print('\n\n***** MATRIZ DE CONFUSAO *****\n')

# final_result = {"question": questions, "answer": answers, "output": outputs}
# final_df = pd.DataFrame(final_result)


answers_df = pd.DataFrame(answers)
outputs_df = pd.DataFrame(outputs)

reports = []
reports_total = []
for col in outputs_df.columns:
  output_dict = classification_report(outputs_df[col], answers_df[col], output_dict=True)
  reports.append(output_dict['macro avg'])
  
  reports_total.append(output_dict)

final_report = pd.DataFrame(reports)
all_data_report = pd.DataFrame(reports_total)

print(final_report)
#print(classification_report(final_df['answer'], final_df['output'], zero_division=0))

print('Done!')
