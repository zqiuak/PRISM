import torch
from get_vit_weight import _replce_weight


@torch.no_grad()
def load_pt(model, weight_file, ckpt_vit_key='swinViT', model_vit_key='swinViT'):
    ckpt = torch.load(weight_file, map_location="cpu", weights_only=False)
    if 'state_dict' in ckpt:
        weight = ckpt['state_dict']
    else:
        weight = ckpt
    selected_weight = _replce_weight(weight, ckpt_vit_key, model_vit_key) # replace the key to match the model's key

    current_model_dict = model.state_dict()
    new_model_dict = {}
    for k in current_model_dict.keys():

        if model_vit_key in k: # AAA.swinViT.BBBB
            weight_key_suffix = model_vit_key + k.split(model_vit_key)[-1] # swinViT.BBBB

            if weight_key_suffix in selected_weight.keys():
                if (current_model_dict[k].size() == selected_weight[weight_key_suffix].size()):
                    new_model_dict[k] = selected_weight[weight_key_suffix].clone()  
                else:
                    try:
                        new_model_dict[k] = selected_weight[weight_key_suffix].expand_as(current_model_dict[k]).clone()
                        print('[INFO] weight size mismatch:', k, "current size:", current_model_dict[k].size(), "pt size:", selected_weight[weight_key_suffix].size(), "expanded to match current model size")
                    except RuntimeError as e:
                        print('[WARNING] error weight size:', k, "current size:", current_model_dict[k].size(), "pt size:", selected_weight[weight_key_suffix].size(), "expand size error:", e)
                        new_model_dict[k] = current_model_dict[k].clone()
            else:
                print('current model weight:', k, "not found in pretrained weight")
                new_model_dict[k] = current_model_dict[k].clone()
                

        else:
            # other weights
            new_model_dict[k] = current_model_dict[k].clone()

    model.load_state_dict(new_model_dict, strict=True)
    return model


@torch.no_grad()
def adapt_weight(model, weights):
    """
    Adapt the model weights to match the given weights.
    :param model: The model to adapt.
    :param weights: The weights to adapt to.
    :return: The adapted model.
    """
    current_model_dict = model.state_dict()
    new_model_dict = {}
    
    for k in current_model_dict.keys():
        if k in weights and current_model_dict[k].size() == weights[k].size():
            new_model_dict[k] = weights[k].clone()
        else:
            print(f'[WARNING] weight mismatch for {k}: current size {current_model_dict[k].size()}, expected size {weights.get(k, "not found")}')
            new_model_dict[k] = current_model_dict[k].clone()

    model.load_state_dict(new_model_dict, strict=True)
    return model