from glob import glob
from tqdm import tqdm
import shutil
import argparse
parser = argparse.ArgumentParser()
import json
from data_loader import SpectrogramDataset, AudioDataLoader, BucketingSampler
import time
import numpy as np
from jiwer import wer
import pickle

def load_data(train_manifest_list,
							valid_manifest_list,
							test_manifest_list, batch_size=12):

  audio_conf = dict(sample_rate=16000,
                  window_size=0.02,
                  window_stride=0.01,
                  window='hamming')
  PAD_CHAR = "¶"
  SOS_CHAR = "§"
  EOS_CHAR = "¤"

  labels_path = './drive/My Drive/labels.json'
  with open(labels_path) as label_file:
  	labels = str(''.join(json.load(label_file)))


	# add PAD_CHAR, SOS_CHAR, EOS_CHAR
  labels = PAD_CHAR + SOS_CHAR + EOS_CHAR + labels
  label2id, id2label = {}, {}
  count = 0
  for i in range(len(labels)):
  	if labels[i] not in label2id:
  		label2id[labels[i]] = count
  		id2label[count] = labels[i]
  		count += 1

  train_data = SpectrogramDataset(audio_conf, manifest_filepath_list=train_manifest_list, label2id=label2id, normalize=True, augment=False)
	# print('train_data ', train_data)
	# train_sampler = BucketingSampler(train_data, batch_size=batch_size)
	# print('train_sampler: ', train_sampler)
	# train_loader = AudioDataLoader(
	# 	train_data, num_workers=4, drop_last = True, batch_sampler=train_sampler)
  train_loader = AudioDataLoader(train_data, batch_size=12 ,num_workers=4, drop_last = True)

  valid_loader_list, test_loader_list = [], []
  for i in range(len(valid_manifest_list)):
  	valid_data = SpectrogramDataset(audio_conf, manifest_filepath_list=[valid_manifest_list[i]], label2id=label2id,
  									normalize=True, augment=False)
  	valid_loader = AudioDataLoader(valid_data, num_workers=4, batch_size=batch_size)
  	valid_loader_list.append(valid_loader)

  for i in range(len(test_manifest_list)):
  	test_data = SpectrogramDataset(audio_conf, manifest_filepath_list=[test_manifest_list[i]], label2id=label2id,
  								normalize=True, augment=False)
  	test_loader = AudioDataLoader(test_data, num_workers=4)
  	test_loader_list.append(test_loader)
  print('done')
  return train_loader, valid_loader_list, test_loader_list, id2label

############################################################################################################
#%% Importing
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

#%%## Defining Network
class Net(nn.Module):
    
    def __init__(self,args):
        super(Net, self).__init__()
        #print("init Net start")

        self.args = args
        initrange = 0.01

        self.conv1 = nn.Conv2d(1,32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10))
        # self.conv1.weight.data.uniform_(-initrange, initrange)
        
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1)
        self.hardtan1 = nn.Hardtanh(min_val=0, max_val=20)

        self.conv2 = nn.Conv2d(32,32, kernel_size=(21, 11), stride=(2, 1))
        # self.conv2.weight.data.uniform_(-initrange, initrange)

        self.bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1)
        self.hardtan2 = nn.Hardtanh(min_val=0, max_val=20)
        
        self.drp1 = nn.Dropout(p=0.1, inplace=False)
        self.flat = nn.Flatten()
        # self.linear0 = nn.Linear(in_features=1826, out_features=args.dim_input)

        self.linear1 = nn.Linear(in_features=args.dim_input, out_features=self.args.dim_model)
        # self.linear1.weight.data.uniform_(-initrange, initrange)

        self.layernorm1 = nn.LayerNorm((self.args.dim_model,), eps=1e-05)
        
        self.query_linear = nn.Linear(in_features = self.args.dim_model, out_features = self.args.num_heads*self.args.dim_key)
        # self.query_linear.weight.data.uniform_(-initrange, initrange)

        self.key_linear = nn.Linear(in_features = self.args.num_heads * self.args.dim_key, out_features = self.args.dim_model)
        # self.key_linear.weight.data.uniform_(-initrange, initrange)

        self.value_linear = nn.Linear(in_features = self.args.dim_model, out_features = self.args.num_heads*self.args.dim_value)
        # self.value_linear.weight.data.uniform_(-initrange, initrange)

        self.layernorm2 = nn.LayerNorm((self.args.dim_model,), eps=1e-05,)

        self.linear2 = nn.Linear(in_features = self.args.num_heads* self.args.dim_value, out_features = self.args.dim_model)
        # self.linear2.weight.data.uniform_(-initrange, initrange)

        self.drp2 = nn.Dropout(p=0.1)
        
        self.drp3 = nn.Dropout(p=0.1)
        self.sftmax = nn.Softmax(dim=2)
        
        self.conv3 = nn.Conv1d(self.args.dim_model, self.args.dim_inner, kernel_size=(1,), stride=(1,))
        # self.conv3.weight.data.uniform_(-initrange, initrange)

        self.conv4 = nn.Conv1d(self.args.dim_inner, self.args.dim_model, kernel_size=(1,), stride=(1,))
        # self.conv4.weight.data.uniform_(-initrange, initrange)

        self.drp4 = nn.Dropout(p=0.1)
        self.layernorm3 = nn.LayerNorm((self.args.dim_model,), eps=1e-05)

        self.max_steps = 5000
        
        #print("init Net end")
             
    def Input_Embedding(self,x):
        # print("Input_Embedding start")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hardtan1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.hardtan2(x)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("Input_Embedding end")
        return x
    
    def Positional_Encoding(self,x):
        # print("Positional_Encoding start")
        # print(x.size())
        shape = x.size()
        max_dims = shape[-1]
        
        if max_dims % 2 == 1: max_dims +=1

        p,i = np.meshgrid(np.arange(self.max_steps), np.arange(max_dims//2))
        pos_emb = np.empty((1, self.max_steps, max_dims))

        pos_emb[0, :, ::2] = np.sin(p/10000**(2*i/max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p/10000**(2*i/max_dims)).T

        self.register_buffer('positional_embedding', torch.tensor(pos_emb, dtype=torch.float))
        
        # print(x.size())
        # if torch.isnan(x).any(): print("yes!")
        # print("Positional_Encoding end")
        return x + self.positional_embedding[:, :shape[-2], :shape[-1]].to('cuda')
        
    def Encoder(self,x,num_layers_encoder):
        # print("Encoder start")
        # print(x.size())
        x = self.drp1(x)
        x = x.mean(3)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.layernorm1(x)
        x = self.Positional_Encoding(x)
        x = num_layers_encoder*self.Encoder_Layer(x)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("Encoder end")
        return x
        
    def Encoder_Layer(self,x):
        # print("Encoder_Layer start")
        # print(x.size())
        x = self.MultiHeadAttention(x)
        x = self.POS_FFN(x)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("Encoder_Layer end")
        return x
    
    def MultiHeadAttention(self,x):
        # print("MultiHeadAttention start")
        # print(x.size())
        x = self.query_linear(x)
        x = self.key_linear(x)
        x = self.value_linear(x)
        x = self.ScaledDotProductAttention(x)
        x = self.linear2(x)
        x = self.layernorm2(x)
        x = self.drp2(x)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("MultiHeadAttention end")
        return x
    
    def ScaledDotProductAttention(self,x):
        # print("ScaledDotProductAttention start")
        # print(x.size())
        x = self.drp3(x)
        x = self.sftmax(x)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("ScaledDotProductAttention end")        
        return x
    
    def POS_FFN(self,x):
        # print("POS_FFN start")
        # print(x.size())
        x = self.conv3(x.view(x.size()[1], x.size()[2],x.size()[0]))
        x = self.conv4(x)
        x = self.drp4(x)
        x = x.view(x.size()[2], x.size()[0], x.size()[1])
        x = self.layernorm3(x)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("POS_FFN end")
        return x
    
    def forward(self,x):
        # print("Forward start")
        # print(x.size())
        x = self.Input_Embedding(x)
        x = self.Positional_Encoding(x)
        x = self.Encoder(x, self.args.num_layers_encoder)
        # if torch.isnan(x).any(): print("yes!")
        # print(x.size())
        # print("Forward end")
        return x

############################################################################################
def greedy_decode(vector, id2label):
  vocab = []
  for i in range(len(vector)):
    vocab.append(id2label[abs(vector[i].item())])
  return vocab

############################################################################################
def cer(hyp, target):
  minlen = len(target) if len(hyp)>len(target) else len(hyp)
  c = 0
  for i in range(minlen):
    if hyp[i]==target[i]:
      c+=1
  return c/len(target)

############################################################################################        
#%% Training Step
train_loss_vector = []
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    ctc_loss = torch.nn.CTCLoss(zero_infinity=True).to(device)
    output_sizes = torch.from_numpy(np.ones((args.batch_size,1))).to(device).long()*256
    for batch_idx, sample in enumerate(train_loader):

        inputs, targets, input_percentages, input_sizes, target_sizes = sample
        data = inputs
        target = targets
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        data = data.float()        
        output = model(data)
        output = F.log_softmax(output, 2)
        input_percentages, target_sizes = input_percentages.to(device), target_sizes.to(device)
        
        output = output.view(output.size(2),output.size(1),output.size(0))
        loss = ctc_loss(output, target, output_sizes, target_sizes)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
          train_loss_vector.append(loss.item())
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))

#%%## Testing Step
test_loss_vector = []
wer_calc = []
cer_calc = []

def valid(args, model, device, valid_loader):
    model.eval()
    test_loss = 0
    ctc_loss = torch.nn.CTCLoss(reduction='sum', zero_infinity=True).to(device)
    output_sizes = torch.from_numpy(np.ones((1,1))).to(device).long()*256
    with torch.no_grad():
        for index, sample in enumerate(valid_loader):
          # print(index)
          inputs, targets, input_percentages, input_sizes, target_sizes = sample
          data = inputs
          target = targets
          data, target = data.to(device), target.to(device)
          data = data.float()
          output = model(data)
          output = F.log_softmax(output, 2)
          output = output.view(output.size(2),output.size(1),output.size(0))
          input_percentages, target_sizes = input_percentages.to(device), target_sizes.to(device)
          test_loss += ctc_loss(output, target, output_sizes, target_sizes).item()
          hypothesis_vocab = greedy_decode(output.int(), id2label)
          target_vocab = greedy_decode(target[0][:target_sizes].int(), id2label)

          wer_calc.append(wer(hypothesis_vocab,target_vocab))
          cer_calc.append(cer(hypothesis_vocab,target_vocab))
            
    test_loss /= len(valid_loader.dataset)
    test_loss_vector.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, 1, len(valid_loader.dataset),
        100. * 1 / len(valid_loader.dataset)))

#%% Main
def main2(train_loader, valid_loader_list, test_loader_list,args):
    # Training settings
    #arser = argparse.ArgumentParser(description='Audio')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=10, metavar='N')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N')
    parser.add_argument('--save-model', default=True)
    parser.add_argument('--sample-rate', default=16000)
    parser.add_argument('--window-size', default=0.02)                    
    
    args = parser.parse_args()
    
    hidden_size = int(math.floor( (args.sample_rate * args.window_size) / 2) + 1)
    hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
    hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
    dim_input = hidden_size * 32
    parser.add_argument('--dim-input', default=dim_input)
    parser.add_argument('--hidden-size', default=hidden_size)
    parser.add_argument('--num-heads', default=8)
    parser.add_argument('--dim-model', default=256)
    parser.add_argument('--dim-key', default=64)
    parser.add_argument('--dim-emb', default=256)
    parser.add_argument('--dim-value', default=64)
    parser.add_argument('--dim-inner', default=1024)
    parser.add_argument('--num-layers-encoder', default=4)
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
    parser.add_argument('--batch-size', type=int, default=12, metavar='N')
    parser.add_argument('--warmup_steps', type=int, default=4000, metavar='N')
    
    args = parser.parse_args()
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    
    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    
    # make a network instance
    model = Net(args).to(device)
    
    # train epochs
    ts = time.time()
    for epoch in range(1, args.epochs + 1):
        # configure optimizer
        lr = args.dim_model**(-0.5)*min(epoch**(-0.5), epoch*args.warmup_steps**(-1.5))
        optimizer = optim.Adam(model.parameters(), lr = lr, betas=(0.9, 0.98), eps=1e-9)
        train(args, model, device, train_loader, optimizer, epoch)
        valid(args, model, device, test_loader_list[0])
    te = time.time()
    duration = te - ts
    print(duration)
    with open('train.pickle', 'wb') as f:
      pickle.dump([train_loss_vector, test_loss_vector, wer_calc, cer_calc], f)

    # save the trained model
    if args.save_model:
	    torch.save(model.state_dict(), "audio2text.pt")

############################################################################################################
#%% main
if __name__ == '__main__':
  parser.add_argument('--train-manifest-list', nargs='+', type=str)
  parser.add_argument('--valid-manifest-list', nargs='+', type=str)
  parser.add_argument('--test-manifest-list', nargs='+', type=str)
  args = parser.parse_args()
  train_loader, valid_loader_list, test_loader_list, id2label = load_data(train_manifest_list=args.train_manifest_list,
							valid_manifest_list=args.valid_manifest_list,
							test_manifest_list=args.test_manifest_list, batch_size=12)
  
  main2(train_loader, valid_loader_list, test_loader_list,args)
  