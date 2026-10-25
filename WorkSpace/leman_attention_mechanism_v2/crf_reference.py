"""LEBERT-CRF, equations 2-10 of Liu et al., ACL 2021.

Independent implementation for transformers 4.56; adapter after BERT layer 1.
"""
import torch
from torch import nn
from transformers import BertConfig, BertModel


class LinearCRF(nn.Module):
    """Unconstrained linear-chain CRF; transitions are indexed [previous,next]."""
    def __init__(self, tags):
        super().__init__()
        self.transitions=nn.Parameter(torch.zeros(tags,tags))
        self.start=nn.Parameter(torch.zeros(tags))
        self.end=nn.Parameter(torch.zeros(tags))

    def forward(self, emissions, tags, mask):
        emissions=emissions.float();batch=torch.arange(len(tags),device=tags.device)
        alpha=self.start+emissions[:,0]
        # A rank containing only single-character sentences has zero transition
        # contribution, but still participates in the same DDP gradient reduction.
        truth=self.start[tags[:,0]]+emissions[batch,0,tags[:,0]]+0*self.transitions.sum()
        for i in range(1,emissions.shape[1]):
            update=torch.logsumexp(alpha[:,:,None]+self.transitions[None],dim=1)+emissions[:,i]
            alpha=torch.where(mask[:,i,None],update,alpha)
            truth=truth+(self.transitions[tags[:,i-1],tags[:,i]]+emissions[batch,i,tags[:,i]])*mask[:,i]
        last=tags.gather(1,(mask.sum(1)-1)[:,None]).squeeze(1)
        truth=truth+self.end[last]
        return (torch.logsumexp(alpha+self.end,dim=1)-truth).mean()

    @torch.no_grad()
    def decode(self, emissions, mask):
        emissions=emissions.float();value=self.start+emissions[:,0];history=[]
        for i in range(1,emissions.shape[1]):
            update,previous=(value[:,:,None]+self.transitions[None]).max(1)
            update=update+emissions[:,i]
            value=torch.where(mask[:,i,None],update,value);history.append(previous)
        endings=(value+self.end).argmax(1).tolist();result=[]
        lengths=mask.sum(1).tolist()
        for b,(end,length) in enumerate(zip(endings,lengths)):
            row=[end]
            for i in range(length-2,-1,-1):row.append(int(history[i][b,row[-1]]))
            result.append(row[::-1])
        return result


class LexiconAdapter(nn.Module):
    def __init__(self,hidden,word_dim,dropout,eps):
        super().__init__()
        self.project1=nn.Linear(word_dim,hidden)
        self.project2=nn.Linear(hidden,hidden)
        self.bilinear=nn.Parameter(torch.empty(hidden,hidden))
        nn.init.normal_(self.bilinear,std=.02)
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(hidden,eps=eps)

    def forward(self,chars,words,valid):
        words=self.dropout(self.project2(torch.tanh(self.project1(words))))
        logits=((chars@self.bilinear).unsqueeze(2)*words).sum(-1)
        weights=logits.float().masked_fill(~valid,-1e4).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        context=(weights.unsqueeze(-1)*words).sum(2)
        return self.norm(self.dropout(chars+context))


class LebertCRF(nn.Module):
    def __init__(self,encoder,word_vectors,num_tags,*,pretrained=True,config=None):
        super().__init__()
        if not pretrained:
            config=config if config is not None else BertConfig.from_pretrained(encoder,local_files_only=True)
            config._attn_implementation='eager'
        self.bert=(BertModel.from_pretrained(encoder,add_pooling_layer=False,
                    attn_implementation='eager',local_files_only=True) if pretrained else
                   BertModel(config,add_pooling_layer=False))
        self.word_embeddings=nn.Embedding.from_pretrained(word_vectors,freeze=False,padding_idx=0)
        c=self.bert.config
        self.adapter=LexiconAdapter(c.hidden_size,word_vectors.shape[1],c.hidden_dropout_prob,c.layer_norm_eps)
        self.dropout=nn.Dropout(c.hidden_dropout_prob)
        self.classifier=nn.Linear(c.hidden_size,num_tags)
        self.crf=LinearCRF(num_tags)
        self.adapter.apply(self.bert._init_weights)
        self.classifier.apply(self.bert._init_weights)

    def forward(self,input_ids,token_mask,word_ids,char_mask,labels=None):
        # One original character/unit per BERT token. CLS/SEP are not CRF labels.
        h=self.bert.embeddings(input_ids=input_ids)
        attention=(1.-token_mask[:,None,None,:].to(h.dtype))*torch.finfo(h.dtype).min
        for i,layer in enumerate(self.bert.encoder.layer):
            h=layer(h,attention_mask=attention)[0]
            if i==0:h=self.adapter(h,self.word_embeddings(word_ids),word_ids.ne(0))
        emissions=self.classifier(self.dropout(h[:,1:-1]))
        if labels is None:return emissions
        with torch.autocast(device_type=emissions.device.type,enabled=False):
            loss=self.crf(emissions.float(),labels,char_mask)
        return loss,emissions
