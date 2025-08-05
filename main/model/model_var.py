
class Model_Var():
    def __init__(self, backbone, scale, modules:str):
        self.backbone = backbone
        if backbone == 'SwinUNETR':
            if scale == 'B':
                self.emb_dim = 48
                self.sw_layer = (2, 2, 2, 2)
            if scale == 'L':
                self.emb_dim = 96
                self.sw_layer = (2, 2, 6, 2)
            if scale == 'H':
                self.emb_dim = 192
                self.sw_layer = (2, 2, 18, 2)
            self.num_heads=(3, 6, 12, 24)
            self.latent_dim = self.emb_dim*16
        elif backbone == 'ResNet':
            self.emb_dim = 512
            self.latent_dim = self.emb_dim*4

        self.encoder_only = 'E' in modules
        self.disentangle = 'D' in modules
        self.pred_meta = 'M' in modules
        self.pred_part = 'P' in modules
        self.synthesize = 'S' in modules
        self.replacew = 'R' in modules
        self.contrastive = 'C' in modules
        self.mix_decoder = 'X' in modules
        self.discriminate = 'I' in modules
        if self.synthesize:
            assert self.disentangle, 'Disentanglement must be used with synthesis'
        if self.mix_decoder:
            assert self.synthesize, 'Mix decoder must be used with synthesis'
        if self.replacew:
            assert self.synthesize, 'Replace weights must be used with synthesis' 
        if self.contrastive:
            assert self.disentangle, 'Disentanglement must be used with contrastive'

        self.loss_weights = {
            'recon': 1.0,
            'meta': 1.0,
            'part': 1.0,
            'disc': 1.0,
            'cont': 0.5,
        }


        self.name = f'{backbone}_{scale}_{modules}'
    
    def __str__(self):
        return f'{self.backbone}_{self.scale}_{self.modules}'
            

'''
EDMSCXI
*D****
**M***
*DM***
*DMS**
*DMSC*
*DM*C*

'''

backbone = 'SwinUNETR'
scale = 'B'
'''
modules: 
    D: feature disentangle, 
    M: predict meta, 
    P: predict part,
    S: synthesize, 
    C: contrastive, 
    X: Mod Decoder, 
    R: Replace seq_f with w, 
    I: Discriminator output
'''
models = ''
models += 'M' # predict meta``
models += 'P' # predict part
models += 'I' # discriminator
models += 'S' # synthesize
models += 'C' # contrastive
models += 'X' # mix decoder

# usually, dont touch these
models += 'D' # disentangle
# models += 'E' # encoder only
# models += 'R' # replace seq_f with w

model_config = Model_Var(backbone, scale, models)

