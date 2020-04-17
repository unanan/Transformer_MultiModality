import torch
from util.trainutil import subsequent_mask

class Translator:
    def __init__(self, max_len, start_symbol):
        self.max_len = max_len
        self.start_symbol = start_symbol
    
    def __call__(self, model, src, src_mask = None):
        raise NotImplementedError()
        
    
    
class GreedyTranslator(Translator):
    '''
    self.model = model
    self.max_len = max_len
    self.start_symbol = start_symbol
    '''
    def __call__(self, model, src, src_mask = None):
        N = src.shape(0)
    
        # Encoding src
        memory = model.encode(src, src_mask)
    
        # Decoding
        ys = torch.ones(1, N).fill_(self.start_symbol).long().cuda()  # .type_as(src.data)
        for i in range(self.max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1))  # .type_as(src.data)
            out = model.decode(memory, src_mask, ys, tgt_mask.cuda())
            print(out.shape)
            prob = model.generator(out[:, -1])
            print(prob.shape)
            _, next_word = torch.max(prob, dim=1)
            print(next_word)
            next_word = next_word.data[0]
            # print(prob)
            ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
        return ys
 
   
# Not Tested
class BeamTranslator(Translator):
    def __call__(self):
        pass