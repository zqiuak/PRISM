import argparse
import torch


parser = argparse.ArgumentParser(description="Save pretrained ViT model weights.")
parser.add_argument("--ckpt", type=str, required=True, help="Input checkpoint.")
parser.add_argument("--out", type=str, required=True, help="Output file.")


def run(infile, outfile):
    ckpt = torch.load(infile, map_location="cpu", weights_only=False)
    if 'state_dict' in ckpt:
        weight = ckpt['state_dict']
    else:
        weight = ckpt
    new_weight = _replce_weight(weight)
    torch.save(new_weight, outfile)
    return 

def _replce_weight(weight, ckpt_vit_key='swinViT', model_vit_key='swinViT'):
    '''
    Select the weight of the swinViT from the checkpoint and rename it to match the model's swinViT key.
    '''
    new_weight = {}
    for k, v in weight.items():
        if ckpt_vit_key in k:
            after_key = k.split(f'{ckpt_vit_key}.')[1]
            new_k = f'{model_vit_key}.{after_key}'
            new_weight[new_k] = v
    return new_weight


if __name__ == "__main__":
    import torch

    args = parser.parse_args()

    run(args.ckpt, args.out)

    print('writing new weight to:', args.out)