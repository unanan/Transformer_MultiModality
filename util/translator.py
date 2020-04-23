import torch
from util.common import subsequent_mask

class Translator:
    def __init__(self, dataparallel, max_len, start_symbol):
        self.dataparallel = dataparallel
        self.max_len = max_len
        self.start_symbol = start_symbol
    
    def __call__(self, model, src, src_mask = None):
        raise NotImplementedError()
        
    
class GreedyTranslator(Translator):
    '''
    Parameters need to provide when initializing:
    self.max_len = max_len
    self.start_symbol = start_symbol
    '''
    def __call__(self, model, src, src_mask = None):
        if self.dataparallel:
            encode = model.module.encode
            decode = model.module.decode
            generator = model.module.generator
        else:
            encode = model.encode
            decode = model.decode
            generator = model.generator
        
        N = src.size(0)
    
        # Encoding src
        memory = encode(src, src_mask)
    
        # Decoding
        ys = torch.ones(N, 1).fill_(self.start_symbol).long().cuda()  # .type_as(src.data)
        for i in range(self.max_len - 1):
            trg_mask = subsequent_mask(ys.size(1))  # .type_as(src.data)
            out = decode(memory, src_mask, ys, trg_mask.cuda())

            prob = generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            # next_word = next_word.data[0]  # deprecated
            # ys = torch.cat([ys, torch.ones(N, 1).long().fill_(next_word).cuda()], dim=1)# deprecated
            next_word = torch.reshape(next_word,(N,1))
            ys = torch.cat([ys, next_word], dim=1)
            
        
        return ys
 
   
# TODO: Not Tested
class BeamTranslator(Translator):
    def __call__(self, model, src, src_mask = None):
        pass