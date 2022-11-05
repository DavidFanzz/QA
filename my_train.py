import pandas as pd
import numpy as np
import copy
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import MegatronBertConfig, MegatronBertModel, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.tokenization_utils_base import BatchEncoding
from torch.optim import AdamW
import tqdm as tqdm

class InputFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
    # def __init__(self, input_ids, attention_mask, token_type_ids, label, input_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_examples_to_dataset(train_features, train_labels, tokenizer):
    features = []
    for (_, example) in enumerate(tqdm.tqdm(zip(train_features, train_labels))):
        tokens = example[0]
        label = example[1]
        features.append(
            InputFeatures(input_ids=tokens.input_ids,
                          attention_mask=tokens.attention_mask,
                          token_type_ids=tokens.token_type_ids,
                          label=label,
                        #   input_len=len(tokens.input_ids[0]),
                          ))
    all_input_ids = torch.tensor([f.input_ids.reshape(512).numpy() for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask.reshape(512).numpy() for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids.reshape(512).numpy() for f in features], dtype=torch.long)
    # all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels
    # return batch.all_input_ids, batch.all_attention_mask, batch.all_token_type_ids, batch.all_labels

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        self.model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
        # self.model = MegatronBertModel.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=768, out_features=2, bias=True)
    
    def forward(self, **input):
        output = self.model(**input)
        pooler_output = output['pooler_output']
        pooler_output = self.dropout(pooler_output)
        output = self.classifier(pooler_output).softmax(dim=1)
        return output


class Lawformer(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained("xcjthu/Lawformer")
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(in_features=768, out_features=2, bias=True)
    
    def forward(self, **input):
        output = self.model(**input)
        pooler_output = output['pooler_output']
        pooler_output = self.dropout(pooler_output)
        output = self.classifier(pooler_output).softmax(dim=1)
        return output


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
    

# if __name__ == "__main__":

    train_batch_size = 4
    test_batch_size = 16
    num_train_epochs = 50
    weight_deacy = 0.01
    learning_rate = 5e-2
    device = torch.device('cuda')
    # Load the train and test files
    
    train_features = torch.load("train_features.pt")
    train_labels = torch.load("train_labels.pt")
    # train_features = train_features[:100]
    # train_labels = train_labels[:100]
    train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_labels, test_size=0.1)

    tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
    config = MegatronBertConfig.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
    model = Model()
    model.to(device)
    
    train_dataset = convert_examples_to_dataset(train_features, train_labels, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    
    test_dataset = convert_examples_to_dataset(test_features, test_labels, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)
    # , collate_fn=collate_fn

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_deacy},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if 'classifier.weight' in n], 'weight_decay': weight_deacy},
    #     {'params': [p for n, p in model.named_parameters() if 'classifier.bias' in n], 'weight_decay': 0.0}
    # ]
    Criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = len(train_dataloader) * num_train_epochs, eta_min=0)
    

    for i in range(int(num_train_epochs)):
        print("epoch:" + str(i))
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
            labels = batch[3]
            inputs = BatchEncoding(inputs)
            output = model(**inputs)
            loss = Criterion(output, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar(step, {'loss': loss.item()})
        torch.save(model, "model" + str(i) + ".pt")

        # true = 0
        # total = 0
        # for step, batch in tqdm.tqdm(enumerate(test_dataloader)):
        #     model.eval()
        #     batch = tuple(t.to(device) for t in batch)
        #     inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'token_type_ids': batch[2]}
        #     labels = batch[3]
        #     inputs = BatchEncoding(inputs)
        #     output = model(**inputs)
        #     predicted_class_id = output.argmax(dim=1)
        #     total += labels.shape[0]
        #     true += (predicted_class_id == labels).sum()
        # print(true / total)


# p(x, y)
