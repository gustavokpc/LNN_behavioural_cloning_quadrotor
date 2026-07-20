#!/usr/bin/env python3
from __future__ import annotations
import copy, subprocess, sys
from pathlib import Path
import numpy as np
import torch
C_DIR=Path(__file__).resolve().parent
PROJECT=C_DIR.parents[2]
sys.path.insert(0,str(PROJECT))
from utils.config import load_yaml, resolve_checkpoint, resolve_saved_config, dataset_dims
from utils.model_builder import build_controller_network
from utils.lightning import Lightning_Model
from utils.data import get_norm_vectors
MODEL_NAME='conv_cfc_default_n64_epoch=17_val_loss=0.000326.ckpt'
STATE=np.array([0.3,-0.2,0.1,0.05,-0.03,0.02,0.01,-0.02,0.03,0.1,-0.1,0.05,0.0,0.0,0.0,5500.0,5600.0,5700.0,5800.0,0.01], dtype=np.float32)
def infer(config,name):
    config=copy.deepcopy(config); low=name.lower(); m=config.setdefault('model',{})
    if 'type' not in m:
        if 'gru' in low: m['type']='gru'
        elif 'lstm' in low: m['type']='lstm'
        elif 'ctrnn' in low: m['type']='ctrnn'
        elif 'rnn' in low: m['type']='simplernn'
        elif 'ncp' in low: m['type']='ncp'
        elif 'cfc' in low: m['type']='cfc'
        elif 'ltc' in low: m['type']='ltc'
        elif 'mlp' in low: m['type']='mlp'
    if 'cfc_pure' in low or '_pure_' in low:
        m['cfc_mode']='pure'; m.setdefault('backbone_units',128)
    return config
def build_c():
    subprocess.run(['gcc','-std=c99','-Wall','-Wextra','-pedantic',str(C_DIR/'run_controller.c'),str(C_DIR/'nn_operations.c'),str(C_DIR/'nn_parameters.c'),'-lm','-o',str(C_DIR/'run_controller')], check=True)
def py_out():
    cfg=infer(load_yaml(resolve_saved_config(MODEL_NAME, PROJECT/'configs')), MODEL_NAME)
    inp,out=dataset_dims(cfg); net=build_controller_network(cfg,inp,out); mod=Lightning_Model(net,cfg)
    mod.load_state_dict(torch.load(resolve_checkpoint(MODEL_NAME,PROJECT),map_location='cpu',weights_only=True)['state_dict']); net.eval()
    mn,mx=get_norm_vectors(cfg['dataset']['input_labels']); x=(STATE-mn.reshape(-1).astype(np.float32))/(mx.reshape(-1).astype(np.float32)-mn.reshape(-1).astype(np.float32)+np.float32(1e-10))
    with torch.no_grad():
        xt=torch.tensor(x.reshape(1,1,1,-1) if cfg.get('conv_block',{}).get('value',False) else x.reshape(1,1,-1), dtype=torch.float32)
        y=net(xt)[0].reshape(-1).numpy()
    return np.clip(y,0,1)
def c_out():
    res=subprocess.run([str(C_DIR/'run_controller')]+[f'{float(v):.9g}' for v in STATE], check=True, capture_output=True, text=True)
    return np.fromstring(res.stdout, sep=' ', dtype=np.float32)
if __name__=='__main__':
    build_c(); yp=py_out(); yc=c_out(); err=np.abs(yp-yc)
    print('python:', yp); print('c:', yc); print('abs error:', err); print('max abs error:', float(err.max()))
