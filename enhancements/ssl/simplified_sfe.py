#!/usr/bin/env python3

from .simplified_byol import (
    SimpleBYOLHead,
    USHEDBYOL,
    create_byol_enhanced_model
)

def create_sfe_enhanced_model(ushed_model, config_overrides=None):
    if config_overrides is None:
        config_overrides = {}
    
    byol_config = {}
    if 'sfe_weight' in config_overrides:
        byol_config['byol_weight'] = config_overrides.pop('sfe_weight')
    if 'ema_decay' in config_overrides:
        byol_config['ema_decay'] = config_overrides['ema_decay']
    if 'proj_dim' in config_overrides:
        byol_config['proj_dim'] = config_overrides['proj_dim']
    if 'detach_target' in config_overrides:
        byol_config['detach_target'] = config_overrides['detach_target']
    
    byol_config.update(config_overrides)
    
    return create_byol_enhanced_model(ushed_model, byol_config)

__all__ = [
    'SimpleBYOLHead',
    'USHEDBYOL',
    'create_byol_enhanced_model',
    'create_sfe_enhanced_model'
]
