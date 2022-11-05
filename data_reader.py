import torch
import json
import tqdm as tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

def ProcessTrainingData(train_file, tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")):
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    train_data = json.load(open(train_file))
    train_questions = []
    train_answers = []
    train_labels = []
    train_features = []
    for i in tqdm.tqdm(range(len(train_data))):
        for key in train_data[i]['option_list'].keys():
            train_question = train_data[i]['statement']
            train_answer = train_data[i]['option_list'][key]
            if key in train_data[i]['answer']:
                train_label = 1
            else:
                train_label = 0
            len_train_question = len(train_question)
            len_train_answer = len(train_answer)
            if len_train_question + len_train_answer > 509:
                train_question = train_question[-(509 - len_train_answer):]
                len_train_questions = 509 - len_train_answer
            encoding = tokenizer([train_question], [train_answer], return_tensors="pt",  padding='max_length', max_length=512)
            train_features.append(encoding)
            train_labels.append(torch.tensor(train_label))
    return train_features, train_labels


# 208 / 82488 的数据tokens > 510
if __name__ == "__main__":
    train_features, train_labels = ProcessTrainingData("/home/scf22/dpfile/CAIL2022/sfks/baseline/dataset/sfks_1st_stage/train.json")
    
    torch.save(train_features, "train_features.pt")
    # torch.save(train_answers, "train_answers.pt")
    torch.save(train_labels, "train_labels.pt")