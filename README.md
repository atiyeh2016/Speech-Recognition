# Speech-Recognition
The aim is to implement the speech recognistion task with an end-to-end model using the voxforge dataset.
This is done by a transformer nertwrok. In these networks the recurrence layers is removed and feedforward layers and a new attension model is used.
First, the data is preproceed using the spectrogram. The implemented transformer network modules are as follow:  
  &emsp;❖ Input Embedding  
  &emsp;❖ Positional Encoding  
  &emsp;❖ Encoder  
 
To embed input data, the following network structure is used.  
    &emsp;(0): Conv2d(1,32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10))  
    &emsp;(1): BatchNorm2d(32, eps=1e-05, momentum=0.1)  
    &emsp;(2): Hardtanh(min_val=0, max_val=20)  
    &emsp;(3): Conv2d(32,32, kernel_size=(21, 11), stride=(2, 1))  
    &emsp;(4): BatchNorm2d(32, eps=1e-05, momentum=0.1)  
    &emsp;(5): Hardtanh(min_val=0, max_val=20)  
  
The positional encoding part is implemneted like the paper "Attention Is All You Need" [1].

The Encoder section is as follows:  
    &emsp;(1): Dropout(p=0.1, inplace=False)  
    &emsp;(2): Linear(in_features=dim_input, out_features=dim-model)  
    &emsp;(3): LayerNorm((dim-model,), eps=1e-05)  
    &emsp;(4): PositionalEncoding()  
    &emsp;(5) (num-layers-encoder) * Encoder Layer  
 
The Encoder Layer consists of two parts:  
  &emsp;(1) Self Attention  
  &emsp;(2) POS_FFN  
  
The structure of Self Attention, POS_FFN, and Attention are as follows:  

 Self Attention:  
   MultiHeadAttention(  
    &emsp;(1): (query_linear): Linear(in_features=dim-model, out_features=num_heads*dim_key)  
    &emsp;(2): (key_linear): Linear(in_features=dim-model, out_features=num_heads * dim_key)  
    &emsp;(3): (value_linear): Linear(in_features=dim-model,out_features=num_heads*dim_value)  
    &emsp;(4): (Attention): ScaledDotProductAttention()  
    &emsp;(5): LayerNorm((dim-model,), eps=1e-05,)  
    &emsp;(6): Linear(in_features=num_heads* dim_value, out_features=dim-model)  
    &emsp;(7): (dropout): Dropout(p=0.1))  
    
  POS_FFN:  
    PositionwiseFeedForwardWithConv(  
    &emsp;(1): Conv1d(dim-model, dim-inner, kernel_size=(1,), stride=(1,))  
    &emsp;(2): Conv1d(dim-inner, dim-model, kernel_size=(1,), stride=(1,))  
    &emsp;(3): Dropout(p=0.1)  
    &emsp;(4): LayerNorm((dim-model,), eps=1e-05))  
    
  Attention:  
    ScaledDotProductAttention(  
    &emsp;(1): (dropout): Dropout(p=0.1)  
    &emsp;(2): (softmax): Softmax(dim=2))  
  
 
[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
