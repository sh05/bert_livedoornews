import numpy as np
import time
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import optim, cuda
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate


DATASET_PATH = './text/livedoor.tsv'
PRETRAINED = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# Load data
df = pd.read_csv(DATASET_PATH, sep='\t')
# print(df.head())

# Split data
categories = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

train, valid_test = train_test_split(
    df,
    test_size=0.2,
    shuffle=True,
    stratify=df[categories]
)
valid, test = train_test_split(
    valid_test,
    shuffle=True,
    test_size=0.5,
    stratify=valid_test[categories]
)

train.reset_index(drop=True, inplace=True)
valid.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# Check data
table = [
        ['train'], [train[category].sum() for category in categories],
        ['valid'], [valid[category].sum() for category in categories],
        ['test'], [test[category].sum() for category in categories]
]

headers = ['data'] + categories
# print(tabulate(table, headers, tablefmt='grid'))

# Define dataset


class NewsDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_len):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        text = self.X[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'labels': torch.Tensor(self.y[index])
        }


MAX_LEN = 128
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

# Create dataset
dataset_train = NewsDataset(
    train['article'], train[categories].values, tokenizer, MAX_LEN)
dataset_valid = NewsDataset(
    valid['article'], valid[categories].values, tokenizer, MAX_LEN)
dataset_test = NewsDataset(
    test['article'], test[categories].values, tokenizer, MAX_LEN)

# for var in dataset_train[0]:
    # print(f'{var}: {dataset_train[0][var]}')


# Define model
class BERTClass(torch.nn.Module):
    def __init__(self, pretrained_model_name, drop_rate, ouput_size):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.drop = torch.nn.Dropout(drop_rate)
        self.fc = torch.nn.Linear(768, ouput_size)

    def forward(self, ids, mask):
        _, out = self.bert(ids, attention_mask=mask, return_dict=False)
        out = self.fc(self.drop(out))
        return out


# Define util function
def calculate_loss_and_accuracy(model, loader, device, criterion=None):
    model.eval()
    loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for data in loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            outputs = model(ids, mask)
            if criterion is not None:
                loss += criterion(outputs, labels).item()

            pred = torch.argmax(outputs, dim=-1).cpu().numpy()
            labels = torch.argmax(labels, dim=-1).cpu().numpy()
            total += len(labels)
            correct += (pred == labels).sum().item()

    return loss / len(loader), correct / total


# Define training function
def train_model(
        dataset_train,
        dataset_valid,
        batch_size,
        model,
        criterion,
        optimizer,
        num_epochs,
        device=None
):
    model.to(device)
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_valid = DataLoader(
        dataset_valid, batch_size=len(dataset_valid), shuffle=False)

    log_train = []
    log_valid = []
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}/{num_epochs}  start!')
        s_time = time.time()

        # Train
        model.train()

        for i, data in enumerate(dataloader_train):
            print(f'\r{i+1}/{len(dataloader_train)}', end='')
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            labels = data['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(ids, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Valid
        loss_train, acc_train = calculate_loss_and_accuracy(
            model, dataloader_train, device, criterion=criterion)
        loss_valid, acc_valid = calculate_loss_and_accuracy(
            model, dataloader_valid, device, criterion=criterion)
        log_train.append([loss_train, acc_train])
        log_valid.append([loss_valid, acc_valid])

        torch.save(
            {'epoch': epoch+1, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()},
            f'checkpoint{epoch+1}.pt'
        )

        e_time = time.time()

        print(f"epoch: {epoch + 1}")
        print(f"loss_train: {loss_train:.4f}")
        print(f"acc_train: {acc_train:.4f},")
        print(f"loss_valid: {loss_valid:.4f}")
        print(f"acc_valid: {acc_valid:.4f}")
        print(f'{(e_time - s_time):.4f}sec')

    return {'train': log_train, 'valid': log_valid}


# define hyperparameters
DROP_RATE = 0.4
OUTPUT_SIZE = len(categories)
# BATCH_SIZE = 4
BATCH_SIZE = 32
# NUM_EPOCHS = 1
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5

# define model
model = BERTClass(PRETRAINED, DROP_RATE, OUTPUT_SIZE)

# define loss function
criterion = torch.nn.BCEWithLogitsLoss()

# define optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

# specify device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cpu')

# train model
log = train_model(
    dataset_train,
    dataset_valid,
    BATCH_SIZE,
    model,
    criterion,
    optimizer,
    NUM_EPOCHS,
    device=device
)


# plot learning curve
x_axis = [x for x in range(1, len(log['train']) + 1)]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(x_axis, np.array(log['train']).T[0], label='train')
ax[0].plot(x_axis, np.array(log['valid']).T[0], label='valid')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].legend()
ax[1].plot(x_axis, np.array(log['train']).T[1], label='train')
ax[1].plot(x_axis, np.array(log['valid']).T[1], label='valid')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('accuracy')
ax[1].legend()
plt.show()
