import yaml
import os
from easydict import EasyDict


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))

def log_cfg_info(cfg, cfg_str_list, logger):
    for key, val in cfg.items():    
        if isinstance(cfg[key], EasyDict):
            cfg_str_list.append('{:30}'.format(key))
            logger.info('{:30} '.format(key))
            log_cfg_info(cfg[key], cfg_str_list, logger)
            continue
        logger.info('{:30}: {}'.format(key, val))
        cfg_str_list.append('{:30}: {}'.format(key, val))

def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value

def get_abs_dirname(file):
    abs_path = os.path.abspath(file)
    dir_name, fn = os.path.split(abs_path)
    return dir_name    
import pdb
def merge_new_config_org(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config_org(config[key], val)
    return config

def merge_new_config(config, new_config, ref_dir=None):
    if '_BASE_CONFIG_' in new_config:
        if (not os.path.exists(new_config['_BASE_CONFIG_'])) and (ref_dir is not None):
            new_config['_BASE_CONFIG_'] = os.path.join(ref_dir, new_config['_BASE_CONFIG_'])
            ref_dir = get_abs_dirname(new_config['_BASE_CONFIG_'])
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        # config.update(EasyDict(yaml_config))
        merge_new_config(config, EasyDict(yaml_config), ref_dir)

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val, ref_dir)
    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
        
        ref_path = get_abs_dirname(cfg_file)
        merge_new_config(config=config, new_config=new_config, ref_dir=ref_path)

    return config

cfg = EasyDict()
# NOTE: ROOT_DIR is given under specific modules
# cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
