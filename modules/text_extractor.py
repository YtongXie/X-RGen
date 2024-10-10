import torch
import torch.nn as nn
from transformers import AutoModel

class MedCLIPTextModel(nn.Module):
    def __init__(self,
        bert_type="emilyalsentzer/Bio_ClinicalBERT",
        proj_dim = 512,
        proj_bias = False) -> None:
        super().__init__()
        self.bert_type = bert_type
        self.last_n_layer = 4
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.projection_head = nn.Linear(768, proj_dim, bias=proj_bias)

    def forward(self, input_ids, attention_mask, topics=False):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if topics:
            return output['hidden_states'][-1]
        else:
            # get 1+2+last layer
            last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
            embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) # pooling
            embed = self.projection_head(embed)
            return embed

