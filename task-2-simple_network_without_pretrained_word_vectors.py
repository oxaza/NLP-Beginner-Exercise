# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchtext import data
from torchtext.data import Field, TabularDataset, BucketIterator
import matplotlib.pyplot as plt
import time

fix_length = 40
# 创建Field对象
TEXT = Field(sequential = True, lower=True, batch_first=True, fix_length = fix_length)
LABEL = Field(sequential = False, use_vocab = False)

# 从文件中读取数据
fields = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]

dataset, test = TabularDataset.splits(path = './', format = 'tsv',
                                      train = 'train.tsv', test = 'test.tsv',
                                      skip_header = True, fields = fields)
train, vali = dataset.split(0.7)

# 构建词表
TEXT.build_vocab(train, max_size=10000, min_freq=10)
LABEL.build_vocab(train)

# 生成向量的批数据
bs = 64
train_iter, vali_iter = BucketIterator.splits((train, vali), batch_size = bs, 
                                              device= torch.device('cpu'), 
                                              sort_key=lambda x: len(x.Phrase),
                                              sort_within_batch=False,
                                              shuffle = True,
                                              repeat = False)

#batch = next(iter(train_iter))
#batch = next(iter(test_iter))
#text, target = batch.Phrase, batch.Sentiment

# 创建网络模型
class EmbNet(nn.Module):
    def __init__(self,emb_size,hidden_size1,hidden_size2=800):
        super().__init__()
        self.embedding = nn.Embedding(emb_size,hidden_size1)
        self.fc = nn.Linear(hidden_size2, 5)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self,x):
        embeds = self.embedding(x).view(x.size(0),-1)
        out = self.fc(embeds)
        out = self.softmax(out)
        return out

model = EmbNet(len(TEXT.vocab.stoi), 20, 20*fix_length)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
def fit(epoch, model, data_loader, phase = 'training'):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
    running_loss = 0.0
    running_correct = 0.0
            
    for batch_idx, batch in enumerate(data_loader):
        text, target = batch.Phrase, batch.Sentiment
#        if torch.cuda.is_available():
#            text, target = text.cuda(), target.cuda()
        if phase == 'training':
            optimizer.zero_grad()
        output = model(text)
        
        loss = F.cross_entropy(output, target)
        
        running_loss += F.cross_entropy(output, target, reduction='sum').item()
        preds = output.data.max(dim=1, keepdim = True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    
    print(f'{phase} loss is {loss:{5}.{4}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    return loss, accuracy

train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]

t0 = time.time()

for epoch in range(1, 100):
    print('epoch no. {} :'.format(epoch) + '-'* 15)
    epoch_loss, epoch_accuracy = fit(epoch, model, train_iter,phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, vali_iter,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

tend = time.time()
print('总共用时：{} s'.format(tend-t0))

plt.figure()
plt.plot(range(1,len(train_losses)+1),train_losses,'b',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()

plt.figure()
plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'b',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
plt.legend()



