"""Fixed-weight full-dev interventions and probes; contains no optimizer or backward."""
import argparse,json,gc
from pathlib import Path
import torch
import torch.distributed as dist
from config import Config
from model import LemanGP,transfer_parent
from data import entity_types,load_split,sha256,write_json
from train import prepare_resources
from controlled_evaluate import evaluate
from probes import probe
from runtime import setup,rank,seed_everything,broadcast_object,input_identity

DEFAULT=['F','no_lexicon','no_attention','dot_attention','no_char','no_boundary','no_recurrence','no_bmes','uniform_lexicon','fixed_gate','no_rope','max_words3','max_words5']
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/root/data/WorkSpace');p.add_argument('--variants',nargs='+',default=DEFAULT)
    a=p.parse_args();device=setup()
    try:
        for v in a.variants:
            c=Config.make(a.root,v,42);c.workers=0
            out=Path(a.root)/'output/leman_short_mechanism_v1/fixed_interventions'/v
            out.mkdir(parents=True,exist_ok=True)
            identity=broadcast_object(input_identity(c) if rank()==0 else None)
            types=entity_types(Path(c.data_dir)/'label.txt')
            paths=broadcast_object(prepare_resources(c,identity,types) if rank()==0 else None)
            dev=load_split(paths[1]);lex=json.loads((Path(c.resource_dir)/'lexicon.json').read_text())
            assert sha256(c.warm_start)==c.warm_start_sha256
            parent=torch.load(c.warm_start,map_location='cpu',mmap=True,weights_only=False)
            seed_everything(c.seed)
            m=LemanGP(c,len(types),len(lex['words']),char_vocab_size=len(lex['characters']),encoder_config=parent['encoder_config'],pretrained=False)
            record=transfer_parent(m,parent['model']);del parent;m.to(device)
            metrics=evaluate(m,dev,c,device,output_dir=out)
            probe(m,dev,c,device,out,'fixed_parent_intervention')
            if rank()==0:
                write_json(out/'IDENTITY.json',{'parent_sha256':c.warm_start_sha256,'transfer':record,'config':c.to_dict(),'training_updates':0})
                print(v,metrics['micro'],flush=True)
            from evaluate import release_loader
            release_loader(dev);del m,dev;gc.collect();torch.cuda.empty_cache()
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
