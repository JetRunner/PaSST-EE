import torch
import collections


def get_net_state_dict_from_checkpoint(ckpt_path):
    state_dict = torch.load(ckpt_path)['state_dict']
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("net."):
            new_state_dict[k[4:]] = v
    return new_state_dict

