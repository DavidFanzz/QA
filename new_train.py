import torch
import torch.nn as nn
import json
import tqdm as tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.optim import AdamW
from transformers import get_scheduler


def ProcessTrainingData(train_file):
    train_data = json.load(open(train_file))
    processed_data = []
    for i in tqdm.tqdm(range(len(train_data))):
        train_question = train_data[i]['statement']
        train_answers = []
        # train_labels = []
        j = 1
        train_label = 0
        for key in train_data[i]['option_list'].keys():
            # train_question = train_data[i]['statement']
            train_answer = train_data[i]['option_list'][key]
            if key in train_data[i]['answer']:
                train_label += j
            j *= 2
            train_answers.append(train_answer)
            # train_labels.append(train_label)
        processed_data.append([train_question, train_answers, torch.tensor(train_label)])
    return processed_data

def ProcessTestData(test_file):
    test_data = json.load(open(test_file))
    processed_data = []
    for i in tqdm.tqdm(range(len(test_data))):
        test_question = test_data[i]['statement']
        test_answers = []
        test_id =  test_data[i]['id']
        for key in test_data[i]['option_list'].keys():
            test_answer = test_data[i]['option_list'][key]
            test_answers.append(test_answer)
        processed_data.append([test_question, test_answers, test_id])
    return processed_data
    # processed_data = []
    # for i in tqdm.tqdm(range(len(train_data))):
    #     train_question = train_data[i]['statement']
    #     train_answers = []
    #     train_labels = []
    #     for key in train_data[i]['option_list'].keys():
    #         # train_question = train_data[i]['statement']
    #         train_answer = train_data[i]['option_list'][key]
    #         if key in train_data[i]['answer']:
    #             train_label = 1
    #         else:
    #             train_label = 0
    #         train_answers.append(train_answer)
    #         train_labels.append(train_label)
    #     processed_data.append([train_question, train_answers, torch.tensor(train_labels)])
    # return processed_data



class Lawformer(nn.Module):
    def __init__(self) -> None:
        super(Lawformer, self).__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("xcjthu/Lawformer")
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=768, out_features=2, bias=True)
        self.mutichoice = nn.Linear(in_features=8, out_features=16, bias=True)
    
    def forward(self, **input):
        output = self.model(**input)
        pooler_output = output['pooler_output']
        pooler_output = self.dropout(pooler_output)
        output = self.classifier(pooler_output)
        new_shape = torch.Size([output.shape[0] // 4, output.shape[1] * 4])
        output = output.reshape(new_shape)
        mutichoice = self.mutichoice(output).softmax(dim=1)
        return mutichoice

import time
class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''
    def __init__(self, n_total,width=30,desc = 'Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current< self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


if __name__ == '__main__':
    train_batch_size = 1
    test_batch_size = 1
    num_train_epochs = 10
    weight_deacy = 0.01
    learning_rate = 1e-5 
    device = torch.device('cuda')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = Lawformer()
    model.to(device)
    
    processed_data = ProcessTrainingData("/home/scf22/dpfile/CAIL2022/sfks/baseline/dataset/sfks_1st_stage/train.json")
    train_data, test_data = train_test_split(processed_data, test_size=0.1)
    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=test_batch_size)



    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_deacy},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    Criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_scheduler(
        name="linear",
        optimizer = optimizer,
        num_warmup_steps = len(train_dataloader) * num_train_epochs * 0.1,
        num_training_steps =  len(train_dataloader) * num_train_epochs,
    )

    for i in range(int(num_train_epochs)):
        print("epoch:" + str(i))
        model.train()
        train_pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            prompts = []
            choices = []
            for i in range(len(batch[2])):
                prompts.extend([batch[0][i] for _ in range(4)])
                choices.extend([batch[1][j][i] for j in range(4)])
            labels = batch[2].reshape(-1).to(device)
            tokens = tokenizer(prompts, choices, return_tensors='pt', padding=True)
            tokens.to(device)
            output = model(**tokens)
            loss = Criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_pbar(step, {'loss': loss.item()})

        test_pbar = ProgressBar(n_total=len(test_dataloader), desc='Testing')
        model.eval()
        true_cnt = 0
        for step, batch in enumerate(test_dataloader):
            prompts = []
            choices = []
            for i in range(len(batch[2])):
                prompts.extend([batch[0][i] for _ in range(4)])
                choices.extend([batch[1][j][i] for j in range(4)])
            labels = batch[2].reshape(-1).to(device)
            tokens = tokenizer(prompts, choices, return_tensors='pt', padding=True)
            tokens.to(device)
            output = model(**tokens)
            predicted_class_id = output.argmax(dim=1)
            true_cnt += ((predicted_class_id.reshape(-1) == labels).sum())
            test_pbar(step)
        print(true_cnt / len(test_dataloader))

    # val_data = ProcessTestData("/home/scf22/dpfile/CAIL2022/sfks/baseline/dataset/sfks_1st_stage/test_input.json")
    # val_sampler = SequentialSampler(val_data)
    # val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=1)
    # val_pbar = ProgressBar(n_total=len(val_dataloader), desc='Valing')
    # model.eval()
    # Note = open('./result.txt', mode='w')
    # for step, batch in enumerate(val_dataloader):
    #     prompts = []
    #     choices = []
    #     answers = ''
    #     id = batch[2]
    #     for i in range(len(batch[2])):
    #         prompts.extend([batch[0][i] for _ in range(4)])
    #         choices.extend([batch[1][j][i] for j in range(4)])
    #     tokens = tokenizer(prompts, choices, return_tensors='pt', padding=True)
    #     tokens.to(device)
    #     output = model(**tokens)
    #     predicted_class_id = output.argmax(dim=1)
    #     if predicted_class_id[0]:
    #         answers += 'A'
    #     if predicted_class_id[1]:
    #         answers += 'B'
    #     if predicted_class_id[2]:
    #         answers += 'C'
    #     if predicted_class_id[3]:
    #         answers += 'D'
    #     Note.writelines([id[0], '\t', answers,'\n'])
    #     val_pbar(step)
    # Note.close()
    # model.to(torch.device('cpu'))
    # torch.save(model.state_dict(), 'model.pt')






    
