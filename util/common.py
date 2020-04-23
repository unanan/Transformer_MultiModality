import json
import numpy as np
import torch

def load_vocabulary(voc_path):
    with open(voc_path, 'r') as jsonfile:
        try:
            jsonfileobj = json.load(jsonfile)
        except:
            raise RuntimeError(f"json file: {voc_path} read failed")
    # jsonfileobj: {"words2index":{"<START>":0,"a":10, }, "index2words":{"0":"<START>", "10":"a", }}
    words2index = jsonfileobj["words2index"]  # {"<START>":0, "a":10, }
    index2words = jsonfileobj["index2words"]  # {"0":"<START>", "10":"a", }
    return words2index,index2words

def statistical_count(values,axis=0):
    ''' Statistical Parameters Calculation: Calculate the avg, median, max, variance. '''
    avg_val = np.mean(values, axis=axis)
    med_val = np.median(values, axis=axis)
    max_val = np.max(values, axis=axis)
    var_val = np.var(values, axis=axis)
    
    return avg_val, med_val, max_val, var_val

#
# def calc_bleus(trg, out, index2words):
#     N = trg.shape(0)
#
#     bleus=[]
#     for i in range(N):
#         reference_str, reference_list = intarr2str(trg[i].cpu().numpy(), index2words)
#         candidate_str, candidate_list = intarr2str(out[i].cpu().numpy(), index2words)
#
#         bleus.append(bleu.count_score(candidates=candidate_list, reference=[reference_list])[0][:4])
#
#

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# def count_score(model_out, ref_tensor, index2words, endid):
#     indices = model_out.max(2)[1].cpu()
#     ref_tensor = ref_tensor.cpu()
#
#     sentence_len = indices.shape[0]
#     scores = 0.0
#     for idx in range(sentence_len):
#         cand_sentence = indices[idx]
#         ref_sentence = ref_tensor[idx]
#
#         # cut the sentence
#         cand_endpos = list(cand_sentence.numpy()).index(endid)
#         ref_endpos = list(ref_sentence.numpy()).index(endid)
#
#         cand_wordlist = [index2words[str(word.numpy())] for word in cand_sentence[:cand_endpos]]
#         ref_wordlist = [index2words[str(word.numpy())] for word in ref_sentence[:ref_endpos]]
#         # print(cand_wordlist)
#         # print(ref_wordlist)
#         # Calculate bleu1-4 average
#         _score = np.average(bleu.count_score(candidates=cand_wordlist,
#                                              reference=[ref_wordlist])[0][:4])
#
#         scores += _score
#
#     return scores / sentence_len
#
#


# def greedy_decode(model, src, max_len, start_symbol, src_mask=None):
#     assert src.shape[0]==1, f"N C H W, N should be 1. {src.shape}"
#
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, 1).fill_(start_symbol).long().cuda()  # .type_as(src.data)
#     for i in range(max_len - 1):
#         tgt_mask = subsequent_mask(ys.size(1))  # .type_as(src.data)
#         out = model.decode(memory, src_mask,
#                            ys,
#                            tgt_mask.cuda())
#         # print(out)
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.data[0]
#         # print(prob)
#         ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
#     return ys


# def batch_greedy_decode(model, src, max_len, start_symbol, src_mask=None):
#     N = src.shape(0)
#
#     memory = model.encode(src, src_mask)
#     ys = torch.ones(1, N).fill_(start_symbol).long().cuda()  # .type_as(src.data)
#     for i in range(max_len - 1):
#         tgt_mask = subsequent_mask(ys.size(1))  # .type_as(src.data)
#         out = model.decode(memory, src_mask,
#                            ys,
#                            tgt_mask.cuda())
#         print(out.shape)
#         prob = model.generator(out[:, -1])
#         print(prob.shape)
#         _, next_word = torch.max(prob, dim=1)
#         print(next_word)
#         next_word = next_word.data[0]
#         # print(prob)
#         ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
#     return ys



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
