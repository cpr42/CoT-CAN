from transformers import AutoModel, AutoConfig
import torch 
import sys
from os.path import join, dirname, abspath
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(torch.nn.Module):
    def __init__(self, model_type, dropout = 0): 
        super(Model, self).__init__()
        model_type = "/data1/chenpr/huggingface/" + model_type # load from local
        self.encoder = AutoModel.from_pretrained(model_type)
        self.config = AutoConfig.from_pretrained(model_type)
        self.encoder_dimension = self.config.hidden_size
        self.soft = torch.nn.Softmax(dim=1)    
        self.drop = torch.nn.Dropout(dropout)
        self.linear = torch.nn.Linear(self.encoder_dimension, 3)
        # 模型没什么特别的，bert加了一个分类器而言
        self.context_attn = torch.nn.MultiheadAttention(
            embed_dim = self.encoder_dimension, num_heads = 8, dropout=0.0, batch_first=True
        )
        self.dataset_attn = torch.nn.MultiheadAttention(
            embed_dim = self.encoder_dimension, num_heads = 8, dropout=0.0, batch_first=True
        )
        # self.prototypes = {}
        

    def forward(self, item):
        last_hidden_states, contexts_embds, event_embds, last_hidden_states_c, attention_mask, attention_mask_c, masked_index = self.get_event_embds(item)
        # torch.Size([100, 9, 768])  
        # b = torch.tensor([[[False if i == 1 else True for i in e ] for e in a]])
        # print("xxx")
        # print(last_hidden_states.size())
        # print(last_hidden_states_c.size())
        # print(attention_mask.size())
        # print(attention_mask_c.size())
        if (last_hidden_states_c != None):
            attn_output, attn_output_weights = self.context_attn(
                query = last_hidden_states, 
                key = last_hidden_states_c, 
                value = last_hidden_states_c,
                key_padding_mask = attention_mask_c) # a True value indicates that the corresponding key value will be ignored for the purpose of attention
            # print("xxx")
            # print(attn_output.size())
            # print(attn_output.size())

            attn_output = attn_output[:,0,:]
            attn_output = attn_output + event_embds

            # attn_output = attn_output + last_hidden_states
            # print(attn_output.size())
            # attn_output = torch.mean(attn_output, dim=1)
            # mask_embeds = []
            # for i in range(last_hidden_states_c.size(0)):
            #     mask_embeds.append(last_hidden_states_c[i, masked_index[i], :])
            # mask_embeds = torch.stack(mask_embeds)
            # attn_output = attn_output + event_embds + mask_embeds
            # attn_output = attn_output.squeeze(1)
            # attn_output = self.soft(attn_output)
            # print(attn_output)
            # logits = self.linear(event_embds)
            logits = self.linear(attn_output)
            logits = self.drop(logits)
            return (logits, attn_output)
        else:
            logits = self.linear(event_embds)
            logits = self.drop(logits)
            return (logits, event_embds)
        
    def forward2(self, item, labeled_representations_centers, original_set):
        last_hidden_states, contexts_embds, event_embds, last_hidden_states_c, attention_mask, attention_mask_c, masked_index = self.get_event_embds(item)
        if (last_hidden_states_c != None):
            attn_output, attn_output_weights = self.context_attn(
                query = last_hidden_states, 
                key = last_hidden_states_c, 
                value = last_hidden_states_c,
                key_padding_mask = attention_mask_c) # a True value indicates that the corresponding key value will be ignored for the purpose of attention
            # print("xxx")
            # print(attn_output.size())
            # print(attn_output.size())

            attn_output = attn_output[:,0,:]
            attn_output = attn_output + event_embds

            # attn_output = attn_output + last_hidden_states
            # print(attn_output.size())
            # attn_output = torch.mean(attn_output, dim=1)
            # mask_embeds = []
            # for i in range(last_hidden_states_c.size(0)):
            #     mask_embeds.append(last_hidden_states_c[i, masked_index[i], :])
            # mask_embeds = torch.stack(mask_embeds)
            # attn_output = attn_output + event_embds + mask_embeds
            # attn_output = attn_output.squeeze(1)
            # attn_output = self.soft(attn_output)
            # print(attn_output)
            # logits = self.linear(event_embds)
        else:
            attn_output = event_embds
        
        # print("xxx")
        events = item["events"]
        labels = item["labels"]
        mask = []
        proto = []
        for i in range(attn_output.size(0)):
            if (events[i] in original_set):
                mask.append(0)
            else:
                mask.append(1)
                proto.append(labeled_representations_centers[labels[i].item()])
        mask = torch.tensor([bool(x) for x in mask])
        # 选择性地处理数据
        data_to_process = attn_output[mask]  # 根据 mask 选择数据
        data_to_keep = attn_output[~mask]    # 保持原样的部分
        # print(data_to_process.unsqueeze(1).size())

        # 对选定部分应用跨注意力
        if data_to_process.size(0) > 0:
            proto = torch.stack(proto).to(device)
            processed_data, attn_output_weights = self.dataset_attn(
                query = data_to_process.unsqueeze(1), 
                key = proto.unsqueeze(1), 
                value = proto.unsqueeze(1),
            )
            processed_data = processed_data.squeeze(1)
            processed_data = processed_data + data_to_process
        else:
            processed_data = data_to_process  # 如果没有数据，则保持原样

        # 将处理后的数据与未处理的数据结合
        combined_output = torch.zeros_like(attn_output)  # 创建一个与 x 形状相同的零张量
        combined_output[mask] = processed_data  # 将处理后的数据放入对应位置
        combined_output[~mask] = data_to_keep  # 将未处理的数据放入对应位置
        # print(combined_output)
        # attn_output, attn_output_weights = self.context_attn(
        #         query = last_hidden_states, 
        #         key = last_hidden_states_c, 
        #         value = last_hidden_states_c,
        #         key_padding_mask = attention_mask_c)
        # print("xxx")
        # for i in range(event_embds.size()):
        logits = self.linear(combined_output)
        logits = self.drop(logits)
        return (logits, attn_output)
        
    def get_event_embds(self, item):
        input_ids = item["event_ids"].to(device)
        input_ids_c = item["contexts_ids"]
        attention_mask = item["event_masks"].to(device) if 'event_masks' in item else None
        if (input_ids_c != None):
            input_ids_c = item["contexts_ids"].to(device)
            attention_mask_c = item["contexts_masks"].to(device) if 'contexts_masks' in item else None
            # print(item["contexts_masks"])
            last_hidden_states, event_embds = self.encoder(input_ids, attention_mask = attention_mask,return_dict = False)
            last_hidden_states_c, contexts_embds = self.encoder(input_ids_c, attention_mask = attention_mask_c,return_dict = False)
            masked_index = [torch.count_nonzero(e).item() - 3 for e in attention_mask_c]
            attention_mask_c = torch.tensor([[False if i == 1 else True for i in e ] for e in attention_mask_c]).to(device)
            # print(attention_mask_c)
        else:
            last_hidden_states, event_embds = self.encoder(input_ids, attention_mask = attention_mask,return_dict = False)
            last_hidden_states_c = None
            attention_mask_c = None
            masked_index = None
            contexts_embds = None
        return last_hidden_states, contexts_embds, event_embds, last_hidden_states_c, attention_mask, attention_mask_c, masked_index # event_embds = [CLS] token
   