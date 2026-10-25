"""Explicit interventions, each named after the computation it removes/replaces."""
import math
import torch
from torch import nn
from architecture_base import LemanGP as Base, initialization_record
from parent_model import ScaledGlobalPointer, boundary_loss
from paper_core import valid_spans

class LemanGP(Base):
    def __init__(self,cfg,num_types,vocab_size,*,char_vocab_size,encoder_config=None,pretrained=True):
        super().__init__(cfg,num_types,vocab_size,char_vocab_size=char_vocab_size,
                         encoder_config=encoder_config,pretrained=pretrained)
        v=cfg.variant
        if v=='no_char':
            for n in ('char_embedding','char_projection','char_scale'):delattr(self,n)
        if v=='no_boundary':del self.boundary
        if v=='no_gru_linear':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed+17003)
                self.pointwise=nn.Linear(self.encoder.config.hidden_size,2*cfg.gru_hidden)
            del self.gru
        if v=='no_recurrence':
            for directions in self.gru.layers:
                for cell in directions:
                    cell.hidden_gates.requires_grad_(False);cell.hidden_candidate.requires_grad_(False)
        if v=='no_bmes':self.adapter.boundary_embedding.requires_grad_(False)
        if v=='uniform_lexicon':self.adapter.bilinear.requires_grad_(False)
        if v=='fixed_gate':self.adapter.gate.requires_grad_(False)
        if v=='gp_reset':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(cfg.seed+17001)
                self.global_pointer=ScaledGlobalPointer(2*cfg.gru_hidden,num_types,cfg.gp_head_size,cfg.max_span)

        self._objective_main=None
        self._objective_parts={}
        def main_loss_hook(module,inputs,result):
            self._objective_main=result.detach()
        def total_loss_hook(module,inputs,result):
            if isinstance(result,tuple) and self._objective_main is not None:
                self._objective_parts={'main':self._objective_main,'boundary_weighted':result[0].detach()-self._objective_main}
        (self.crf if v=='crf' else self.loss_fn).register_forward_hook(main_loss_hook)
        self.register_forward_hook(total_loss_hook)

    def lexicon_forward(self,x,words,bmes):
        a=self.adapter;valid=words.ne(0)
        if self.training:
            keep=torch.rand(a.word_embedding.num_embeddings,device=words.device)>=self.cfg.lexicon_word_dropout
            valid=valid & keep[words]
            valid=valid & (torch.rand(words.shape[0],1,1,device=words.device)>=self.cfg.lexicon_dropout)
        boundary=a.boundary_embedding(bmes)
        if self.cfg.variant=='no_bmes':boundary=torch.zeros_like(boundary)
        c=a.projection(torch.cat([a.word_embedding(words),boundary],-1))
        if self.cfg.variant=='uniform_lexicon':w=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
        else:
            scores=(a.bilinear(x).unsqueeze(2)*c).sum(-1)/math.sqrt(x.shape[-1])
            w=scores.masked_fill(~valid,-1e30).softmax(-1)*valid
        context=(w.unsqueeze(-1)*c).sum(2)
        gate=(torch.full_like(x,.5) if self.cfg.variant=='fixed_gate' else
              torch.sigmoid(a.gate(torch.cat([x,context],-1))))
        y=a.norm(x+a.dropout(gate*context))
        return torch.where(valid.any(-1,keepdim=True),y,x)

    def pointwise_gru(self,x,mask):
        # Set previous recurrent state to zero at every position, both directions.
        # Reuse trained input projections and update gates; no neighbor access remains.
        for i,directions in enumerate(self.gru.layers):
            parts=[]
            for cell in directions:
                p=cell.input_projection(x);h=cell.hidden_candidate.shape[0]
                parts.append(torch.sigmoid(p[...,:h])*torch.tanh(p[...,2*h:]))
            x=torch.cat(parts,-1)*mask.unsqueeze(-1)
            if i+1<len(self.gru.layers):x=self.gru.dropout(x)
        return x

    def forward(self,input_ids,token_mask,char_pool,char_mask,word_ids,boundary_ids,lengths,char_ids,labels=None,crf_tags=None):
        v=self.cfg.variant
        # Exact baseline and pre-existing controlled variants retain their source forward.
        if v in ('F','no_lexicon','no_attention','dot_attention','max_words3','max_words5','crf','encoder_gp','gp_reset','fgm_control','no_fgm','clean_two_pass','weighted_control','unit_loss'):
            if v in ('gp_reset','fgm_control','no_fgm','clean_two_pass','weighted_control','unit_loss'):
                from parent_model import LemanGP as Parent
                return Parent.forward(self,input_ids,token_mask,char_pool,char_mask,word_ids,boundary_ids,lengths,char_ids,labels)
            return super().forward(input_ids,token_mask,char_pool,char_mask,word_ids,boundary_ids,lengths,char_ids,labels,crf_tags)
        hidden=self.encoder(input_ids=input_ids,attention_mask=token_mask).last_hidden_state
        with torch.autocast(device_type=input_ids.device.type,enabled=False):
            x=torch.bmm(char_pool.float(),hidden.float())
            if v!='no_char':x=x+self.char_scale*self.char_projection(self.char_embedding(char_ids))
            if v in ('no_bmes','uniform_lexicon','fixed_gate'):x=self.lexicon_forward(x,word_ids,boundary_ids)
            else:x=self.adapter(x,word_ids,boundary_ids)
            x=x*char_mask.unsqueeze(-1)
            if v=='no_recurrence':sequence=self.pointwise_gru(x,char_mask)
            elif v=='no_gru_linear':sequence=self.pointwise(x)
            else:sequence=self.gru(x,char_mask)
            sequence=self.dropout(self.sequence_norm(sequence))*char_mask.unsqueeze(-1)
            attended=self.attention(sequence,char_mask)
            if v=='no_rope':
                gp=self.global_pointer;b,l,_=attended.shape
                z=gp.projection(attended).reshape(b,l,self.num_types,2,gp.head_size)
                scores=torch.einsum('bith,bjth->btij',z[...,0,:],z[...,1,:])/math.sqrt(gp.head_size)
                pos=torch.arange(l,device=x.device)
                valid=valid_spans(char_mask)&((pos[None,:]-pos[:,None])<gp.max_span)[None,None]
                scores=scores.masked_fill(~valid,-1e30)
            else:scores=self.global_pointer(attended,char_mask)
            if labels is None:return scores
            loss=self.loss_fn(scores,labels,char_mask)
            if v!='no_boundary':
                auxiliary=self.boundary(sequence).reshape(*sequence.shape[:2],self.num_types,2)
                loss=loss+self.cfg.boundary_weight*boundary_loss(auxiliary,labels,char_mask)
            return loss,scores

def transfer_parent(model,state):
    """Load every compatible retained tensor. Reject every unplanned difference."""
    v=model.cfg.variant;target=model.state_dict();loaded={};missing=[];dropped=[]
    reset=('global_pointer.' if v in ('gp_reset','encoder_gp') else None)
    for k,t in target.items():
        if k in state and state[k].shape==t.shape and not (reset and k.startswith(reset)):
            loaded[k]=state[k]
        else:missing.append(k)
    for k in state:
        if k not in loaded:dropped.append(k)
    allowed_new={'crf':('classifier.','crf.'),'gp_reset':('global_pointer.',),
                 'encoder_gp':('global_pointer.',),'no_gru_linear':('pointwise.',)}.get(v,())
    allowed_drop={'no_lexicon':('adapter.',),'no_attention':('attention.',),
       'no_char':('char_embedding.','char_projection.','char_scale'),'no_boundary':('boundary.',),
       'no_gru_linear':('gru.',),'crf':('global_pointer.',),'gp_reset':('global_pointer.',),
       'encoder_gp':('char_embedding.','char_projection.','char_scale','adapter.','gru.','sequence_norm.','attention.','boundary.','global_pointer.')}.get(v,())
    assert all(k.startswith(allowed_new) for k in missing),(v,missing)
    assert all(k.startswith(allowed_drop) for k in dropped),(v,dropped)
    model.load_state_dict(loaded,strict=False)
    return {'loaded_tensors':len(loaded),'loaded_numel':sum(x.numel() for x in loaded.values()),
            'new_or_reset_tensors':missing,'dropped_parent_tensors':dropped,
            'parent_variant':'F','parent_seed':42,'conditional_on_trained_parent':True}
