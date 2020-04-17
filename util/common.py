import numpy as np
import torch
import torch.nn as nn
import util.bleu as bleu                                # validating metrics: bleu (refer to open codes on Github)

#======================================================= Wrappers ======================================================




class MetricWrapper:
    " Validate Metrics wrapper. "

    def __init__(self, index2words, metriccalculator):
        self.index2words = index2words
        self.metriccalculator = metriccalculator


    def intarr2str(self, intarr, index2words):
        outputstr = ""
        outputlist = []
        if len(intarr.shape) > 1:
            intarr = intarr[0]
        for ele in intarr:
            if ele == 2 or ele == 179:
                break
            outputstr += index2words[str(ele)]
            outputlist.append(ele)
        return outputstr, outputlist

    def __call__(self,out, trg):
        assert out.shape==trg.shape, f"out:{out.shape} trg:{trg.shape}, shapes of out and trg must be the same."
        
        
        pass

#======================================================== Utils ========================================================

def split_data(data):
    src, trg, loss_mask, trg_ntokens = data
    src, trg, loss_mask, trg_ntokens = src.cuda(), trg.cuda(), loss_mask.cuda(), trg_ntokens
    return src, trg, loss_mask, trg_ntokens


def calc_metrics(losses,axis=0):
    ''' Calculate the avg, median, max, variance of a numpy array. '''
    avg_loss = np.mean(losses, axis=axis)
    med_loss = np.median(losses, axis=axis)
    max_loss = np.max(losses, axis=axis)
    var_loss = np.var(losses, axis=axis)
    
    return avg_loss, med_loss, max_loss, var_loss


def calc_bleus(trg, out, index2words):
    N = trg.shape(0)
    
    bleus=[]
    for i in range(N):
        reference_str, reference_list = intarr2str(trg[i].cpu().numpy(), index2words)
        candidate_str, candidate_list = intarr2str(out[i].cpu().numpy(), index2words)
        
        bleus.append(bleu.count_score(candidates=candidate_list, reference=[reference_list])[0][:4])
    
    

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def count_score(model_out, ref_tensor, index2words, endid):
    indices = model_out.max(2)[1].cpu()
    ref_tensor = ref_tensor.cpu()
    
    sentence_len = indices.shape[0]
    scores = 0.0
    for idx in range(sentence_len):
        cand_sentence = indices[idx]
        ref_sentence = ref_tensor[idx]
        
        # cut the sentence
        cand_endpos = list(cand_sentence.numpy()).index(endid)
        ref_endpos = list(ref_sentence.numpy()).index(endid)
        
        cand_wordlist = [index2words[str(word.numpy())] for word in cand_sentence[:cand_endpos]]
        ref_wordlist = [index2words[str(word.numpy())] for word in ref_sentence[:ref_endpos]]
        # print(cand_wordlist)
        # print(ref_wordlist)
        # Calculate bleu1-4 average
        _score = np.average(bleu.count_score(candidates=cand_wordlist,
                                             reference=[ref_wordlist])[0][:4])
        
        scores += _score
    
    return scores / sentence_len



        

def greedy_decode(model, src, max_len, start_symbol, src_mask=None):
    assert src.shape(0)==1, f"N C H W, N should be 1. {src.shape}"
    
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()  # .type_as(src.data)
    for i in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1))  # .type_as(src.data)
        out = model.decode(memory, src_mask,
                           ys,
                           tgt_mask.cuda())
        # print(out)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        # print(prob)
        ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
    return ys


def batch_greedy_decode(model, src, max_len, start_symbol, src_mask=None):
    N = src.shape(0)
    
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, N).fill_(start_symbol).long().cuda()  # .type_as(src.data)
    for i in range(max_len - 1):
        tgt_mask = subsequent_mask(ys.size(1))  # .type_as(src.data)
        out = model.decode(memory, src_mask,
                           ys,
                           tgt_mask.cuda())
        print(out.shape)
        prob = model.generator(out[:, -1])
        print(prob.shape)
        _, next_word = torch.max(prob, dim=1)
        print(next_word)
        next_word = next_word.data[0]
        # print(prob)
        ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
    return ys





def visualize(epoch, input):  # C,target,fin1,fin2,fin3):
    from visdom import Visdom
    viz = Visdom()
    assert viz.check_connection()
    
    # print(tar.shape)
    
    # for c in range(C):
    # print(fin1.cpu()[n,c][np.newaxis, :].shape)
    # viz.image(
    #     fin1.cpu()[0,c][np.newaxis, :],
    #     opts=dict(title='fin1', caption='fin1'),
    # )
    # viz.image(
    #     fin2.cpu()[0,c][np.newaxis, :],
    #     opts=dict(title='fin2', caption='fin2'),
    # )
    # viz.image(
    #     fin3.cpu()[0,c][np.newaxis, :],
    #     opts=dict(title='fin3', caption='fin3'),
    # )
    
    viz.heatmap(input[0, 0],
                opts=dict(colormap='Electric', title='Epoch-{} input'.format(epoch)))
    # viz.heatmap(X=target[0, c],
    #             opts=dict(colormap='Electric', title='Epoch-{} Points-{} target'.format(epoch, c)))
    # viz.heatmap(X=fin1[0, c],
    #             opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin1'.format(epoch, c)))
    # viz.heatmap(X=fin2[0, c],
    #             opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin2'.format(epoch, c)))
    # viz.heatmap(X=fin3[0, c],
    #             opts=dict(colormap='Electric', title='Epoch-{} Points-{} fin3'.format(epoch, c)))
    return
