import torch
from torch import nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math
import collections


class cross_attention(nn.Sequential):
    def __init__(self, hidden_dim):
        super(cross_attention, self).__init__()
        transformer_emb_size_drug = hidden_dim
        # transformer_dropout_rate = 0.1
        transformer_n_layer_drug = 4
        transformer_intermediate_size_drug = hidden_dim
        transformer_num_attention_heads_drug = 4
        transformer_attention_probs_dropout = 0.1
        transformer_hidden_dropout_rate = 0.1
        
        self.encoder = Encoder_1d(transformer_n_layer_drug,
                                         transformer_emb_size_drug,
                                         transformer_intermediate_size_drug,
                                         transformer_num_attention_heads_drug,
                                         transformer_attention_probs_dropout,
                                         transformer_hidden_dropout_rate)
    
    def forward(self, emb, ex_e_mask,device1):
        global device
        device = device1

        encoded_layers, attention_scores = self.encoder(emb, ex_e_mask)
        return encoded_layers, attention_scores


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class CrossFusion(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossFusion, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_ligand = nn.Linear(hidden_size, self.all_head_size)
        self.key_ligand = nn.Linear(hidden_size, self.all_head_size)
        self.value_ligand = nn.Linear(hidden_size, self.all_head_size)
        
        self.query_receptor = nn.Linear(hidden_size, self.all_head_size)
        self.key_receptor = nn.Linear(hidden_size, self.all_head_size)
        self.value_receptor = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        
        # ligand
        ligand_hidden = hidden_states[0]
        ligand_mask = attention_mask[0]
        
        # receptor
        receptor_hidden = hidden_states[1]
        receptor_mask = attention_mask[1]
        
        ligand_mask = ligand_mask.unsqueeze(1).unsqueeze(2)
        ligand_mask = ((1.0 - ligand_mask) * -10000.0).to(device)
        
        receptor_mask = receptor_mask.unsqueeze(1).unsqueeze(2)
        receptor_mask = ((1.0 - receptor_mask) * -10000.0).to(device)
      
        mixed_query_layer_ligand = self.query_ligand(ligand_hidden)
        mixed_key_layer_ligand = self.key_ligand(ligand_hidden)
        mixed_value_layer_ligand = self.value_ligand(ligand_hidden)

        query_layer_ligand = self.transpose_for_scores(mixed_query_layer_ligand)
        key_layer_ligand = self.transpose_for_scores(mixed_key_layer_ligand)
        value_layer_ligand = self.transpose_for_scores(mixed_value_layer_ligand)
        
        mixed_query_layer_receptor = self.query_receptor(receptor_hidden)
        mixed_key_layer_receptor = self.key_receptor(receptor_hidden)
        mixed_value_layer_receptor = self.value_receptor(receptor_hidden)
        
        query_layer_receptor = self.transpose_for_scores(mixed_query_layer_receptor)
        key_layer_receptor = self.transpose_for_scores(mixed_key_layer_receptor)
        value_layer_receptor = self.transpose_for_scores(mixed_value_layer_receptor)

        # receptor as query, ligand as key,value
        attention_scores_receptor = torch.matmul(query_layer_receptor, key_layer_ligand.transpose(-1, -2))
        attention_scores_receptor = attention_scores_receptor / math.sqrt(self.attention_head_size)
        attention_scores_receptor = attention_scores_receptor + ligand_mask
        attention_probs_receptor = nn.Softmax(dim=-1)(attention_scores_receptor)
        attention_probs_receptor = self.dropout(attention_probs_receptor)
        
        context_layer_receptor = torch.matmul(attention_probs_receptor, value_layer_ligand)
        context_layer_receptor = context_layer_receptor.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_receptor = context_layer_receptor.size()[:-2] + (self.all_head_size,)
        context_layer_receptor = context_layer_receptor.view(*new_context_layer_shape_receptor)
        
        # ligand as query, receptor as key,value
        attention_scores_ligand = torch.matmul(query_layer_ligand, key_layer_receptor.transpose(-1, -2))
        attention_scores_ligand = attention_scores_ligand / math.sqrt(self.attention_head_size)
        attention_scores_ligand = attention_scores_ligand + receptor_mask
        attention_probs_ligand = nn.Softmax(dim=-1)(attention_scores_ligand)
        attention_probs_ligand = self.dropout(attention_probs_ligand)
        
        context_layer_ligand = torch.matmul(attention_probs_ligand, value_layer_receptor)
        context_layer_ligand = context_layer_ligand.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape_ligand = context_layer_ligand.size()[:-2] + (self.all_head_size,)
        context_layer_ligand = context_layer_ligand.view(*new_context_layer_shape_ligand)
        
        # output of cross fusion
        context_layer = [context_layer_ligand, context_layer_receptor]
        # attention of cross fusion
        attention_probs = [attention_probs_ligand, attention_probs_receptor]

        return context_layer, attention_probs
    

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense_ligand = nn.Linear(hidden_size, hidden_size)
        self.dense_receptor = nn.Linear(hidden_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states_ligand = self.dense_ligand(hidden_states[0])
        hidden_states_ligand = self.dropout(hidden_states_ligand)

        hidden_states_ligand = self.LayerNorm(hidden_states_ligand + input_tensor[0])
        
        hidden_states_receptor = self.dense_receptor(hidden_states[1])
        hidden_states_receptor = self.dropout(hidden_states_receptor)
        hidden_states_receptor = self.LayerNorm(hidden_states_receptor + input_tensor[1])
        return [hidden_states_ligand, hidden_states_receptor]    
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = CrossFusion(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_scores = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_scores    
    
class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense_ligand = nn.Linear(hidden_size, hidden_size)
        self.dense_receptor = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        
        hidden_states_ligand = self.dense_ligand(hidden_states[0])
        hidden_states_ligand = F.relu(hidden_states_ligand)
        
        hidden_states_receptor = self.dense_receptor(hidden_states[1])
        hidden_states_receptor = F.relu(hidden_states_receptor)
        
        return [hidden_states_ligand, hidden_states_receptor]    

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense_ligand = nn.Linear(hidden_size, hidden_size)
        self.dense_receptor = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):

        hidden_states_ligand = self.dense_ligand(hidden_states[0])
        hidden_states_ligand = self.dropout(hidden_states_ligand)
        hidden_states_ligand = self.LayerNorm(hidden_states_ligand + input_tensor[0])
        
        hidden_states_receptor = self.dense_receptor(hidden_states[1])
        hidden_states_receptor = self.dropout(hidden_states_receptor)
        hidden_states_receptor = self.LayerNorm(hidden_states_receptor + input_tensor[1])
        return [hidden_states_ligand, hidden_states_receptor]    
    
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_scores = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_scores    

    
class Encoder_1d(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size,
                 num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_1d, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads,
                        attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])
        
        # 1: ligand_seq; 2: ligand_stru; 3: receptor_seq; 4: receptor_stru
        # self.cls = nn.Embedding(5, hidden_size,padding_idx=0)
        
        # modality embedding; 0 for seq; 1 for stru;
        self.mod = nn.Embedding(2, hidden_size)
        
        # # receptor type embedding; 0 for ligand; 1 for small receptor
        # self.type_e = nn.Embedding(2, 256)
        
    

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # add 1 for mask
        # attention_mask[0] = torch.cat((torch.ones(attention_mask[0].size()[0]).unsqueeze(1).to(device), attention_mask[0]), dim=1).to(device)
        # attention_mask[1] = torch.cat((torch.ones(attention_mask[1].size()[0]).unsqueeze(1).to(device), attention_mask[1]), dim=1).to(device)
        # attention_mask[2] = torch.cat((torch.ones(attention_mask[2].size()[0]).unsqueeze(1).to(device), attention_mask[2]), dim=1).to(device)
        # attention_mask[3] = torch.cat((torch.ones(attention_mask[3].size()[0]).unsqueeze(1).to(device), attention_mask[3]), dim=1).to(device)
        
        # cls_ligand_seq = torch.tensor([1]).expand(hidden_states[0].size()[0],1).to(device)
        # cls_ligand_seq = self.cls(cls_ligand_seq)
        # hidden_states[0] = torch.cat((cls_ligand_seq, hidden_states[0]),dim=1)
        
        # cls_ligand_stru = torch.tensor([2]).expand(hidden_states[1].size()[0],1).to(device)
        # cls_ligand_stru = self.cls(cls_ligand_stru)
        # hidden_states[1] = torch.cat((cls_ligand_stru, hidden_states[1]),dim=1)
       
        # cls_receptor_seq = torch.tensor([3]).expand(hidden_states[2].size()[0],1).to(device)
        # cls_receptor_seq = self.cls(cls_receptor_seq)
        # hidden_states[2] = torch.cat((cls_receptor_seq, hidden_states[2]),dim=1)
        
        # cls_receptor_stru = torch.tensor([4]).expand(hidden_states[3].size()[0],1).to(device)
        # cls_receptor_stru = self.cls(cls_receptor_stru)
        # hidden_states[3] = torch.cat((cls_receptor_stru, hidden_states[3]),dim=1)


        # for seq
        seq_ligand_emb1 = torch.tensor([0]).expand(hidden_states[0].size()[0],hidden_states[0].size()[1]).to(device)
        seq_ligand_emb1 = self.mod(seq_ligand_emb1)
        hidden_states[0] = hidden_states[0] + seq_ligand_emb1
        
        seq_receptor_emb1 = torch.tensor([0]).expand(hidden_states[2].size()[0],hidden_states[2].size()[1]).to(device)
        seq_receptor_emb1 = self.mod(seq_receptor_emb1)
        hidden_states[2] = hidden_states[2] + seq_receptor_emb1
        
        # for stru
        stru_ligand_emb1 = torch.tensor([1]).expand(hidden_states[1].size()[0],hidden_states[1].size()[1]).to(device)
        stru_ligand_emb1 = self.mod(stru_ligand_emb1)
        hidden_states[1] = hidden_states[1] + stru_ligand_emb1
        
        stru_receptor_emb1 = torch.tensor([1]).expand(hidden_states[3].size()[0],hidden_states[3].size()[1]).to(device)
        stru_receptor_emb1 = self.mod(stru_receptor_emb1)
        hidden_states[3] = hidden_states[3] + stru_receptor_emb1
        
        # # for ligand
        # seq_ligand_emb2 = torch.tensor([0]).expand(hidden_states[0].size()[0],hidden_states[0].size()[1]).to(device)
        # seq_ligand_emb2 = self.type_e(seq_ligand_emb2)
        # hidden_states[0] = hidden_states[0] + seq_ligand_emb2
        
        # stru_ligand_emb2 = torch.tensor([0]).expand(hidden_states[2].size()[0],hidden_states[2].size()[1]).to(device)
        # stru_ligand_emb2 = self.type_e(stru_ligand_emb2)
        # hidden_states[2] = hidden_states[2] + stru_ligand_emb2
        
        # # for small receptor
        # seq_receptor_emb2 = torch.tensor([1]).expand(hidden_states[1].size()[0],hidden_states[1].size()[1]).to(device)
        # seq_receptor_emb2 = self.type_e(seq_receptor_emb2)
        # hidden_states[1] = hidden_states[1] + seq_receptor_emb2
        
        # stru_receptor_emb2 = torch.tensor([1]).expand(hidden_states[3].size()[0],hidden_states[3].size()[1]).to(device)
        # stru_receptor_emb2 = self.type_e(stru_receptor_emb2)
        # hidden_states[3] = hidden_states[3] + stru_receptor_emb2
        ligand_hidden = torch.cat((hidden_states[0], hidden_states[1]), dim=1)
        receptor_hidden = torch.cat((hidden_states[2], hidden_states[3]), dim=1)
        
        ligand_mask = torch.cat((attention_mask[0], attention_mask[1]), dim=1)
        receptor_mask = torch.cat((attention_mask[2], attention_mask[3]), dim=1)

        hidden_states = [ligand_hidden, receptor_hidden]        
        attention_mask = [ligand_mask, receptor_mask]
        
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_scores = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
               all_encoder_layers.append(hidden_states)
        return all_encoder_layers, attention_scores
    