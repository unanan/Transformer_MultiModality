import math
import six
from collections import defaultdict


# Functions
def _precook(s, n=4, out=False):
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    if isinstance(s,str):
        words = s.split()
        # print(words)
    elif isinstance(s,list) or isinstance(s,tuple):
        words = s
    else:
        raise NotImplementedError(type(s))
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return (len(words), counts)


def _single_reflen(reflens, option=None, testlen=None):
    if option == "shortest":
        reflen = min(reflens)
    elif option == "average":
        reflen = float(sum(reflens)) / len(reflens)
    elif option == "closest":
        reflen = min((abs(l - testlen), l) for l in reflens)[1]
    else:
        assert False, "unsupported reflen option %s" % option
    
    return reflen


def _cook_refs(refs, eff=None, n=4):  ## lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.'''
    
    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = _precook(ref, n)
        reflen.append(rl)
        for (ngram, count) in six.iteritems(counts):
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)
    
    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen)) / len(reflen)
    
    ## lhuang: N.B.: leave reflen computaiton to the very end!!
    
    ## lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)
    
    return (reflen, maxcounts)


def _cook_test(test, reflen_refmaxcounts, eff=None, n=4):
    '''Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.'''
    
    reflen, refmaxcounts = reflen_refmaxcounts
    testlen, counts = _precook(test, n, True)
    
    result = {}
    
    # Calculate effective reference sentence length.
    
    if eff == "closest":
        result["reflen"] = min((abs(l - testlen), l) for l in reflen)[1]
    else:  ## i.e., "average" or "shortest" or None
        result["reflen"] = reflen
    
    result["testlen"] = testlen
    
    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]
    
    result['correct'] = [0] * n
    for (ngram, count) in six.iteritems(counts):
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)
    
    return result


def _cook_append(test, refs, crefs, ctest):
    '''called by constructor and __iadd__ to avoid creating new instances.'''
    
    if refs is not None:
        crefs.append(_cook_refs(refs))
        if test is not None:
            cooked_test = _cook_test(test, crefs[-1])
            ctest.append(cooked_test)  ## N.B.: -1
        else:
            ctest.append(None)  # lens of crefs and ctest have to match
    return crefs, ctest


# TODO: Need to test
def count_score(candidates, reference):
    # Constant Parameters
    __method__ = 'BLEU'
    n = 4
    option = "average"  # / "closest"
    small = 1e-9
    tiny = 1e-15  ## so that if guess is 0 still return 0
    testlen = 0
    reflen = 0
    verbose = 0
    ctest = []
    crefs = []
    score = None
    

    
    # Variables Init
    bleu_list = [[] for _ in range(n)]
    totalcomps = {'testlen': 0, 'reflen': 0, 'guess': [0] * n, 'correct': [0] * n}
    
    crefs, ctest = _cook_append(candidates, reference, crefs, ctest)
    
    # For each sentence
    for comps in ctest:
        _testlen = comps['testlen']
        testlen += _testlen
        
        _reflen = _single_reflen(comps['reflen'], option, _testlen)
        
        reflen += _reflen
        
        for key in ['guess', 'correct']:
            for k in range(n):
                totalcomps[key][k] += comps[key][k]
        
        # append per image bleu score
        bleu = 1.
        for k in range(n):
            bleu *= (float(comps['correct'][k]) + tiny) \
                    / (float(comps['guess'][k]) + small)
            bleu_list[k].append(bleu ** (1. / (k + 1)))
        ratio = (testlen + tiny) / (reflen + small)  ## N.B.: avoid zero division
        if ratio < 1:
            for k in range(n):
                bleu_list[k][-1] *= math.exp(1 - 1 / ratio)
        
        if verbose > 1:
            pass
            # print(comps, reflen)
    
    totalcomps['reflen'] = reflen
    totalcomps['testlen'] = testlen
    
    bleus = []
    bleu = 1.
    for k in range(n):
        bleu *= float(totalcomps['correct'][k] + tiny) \
                / (totalcomps['guess'][k] + small)
        bleus.append(bleu ** (1. / (k + 1)))
    ratio = (testlen + tiny) / (reflen + small)  ## N.B.: avoid zero division
    if ratio < 1:
        for k in range(n):
            bleus[k] *= math.exp(1 - 1 / ratio)
    
    if verbose > 0:
        pass
        # print(totalcomps)
        # print("ratio:", ratio)
    
    score = bleus
    return score, bleu_list


if __name__ == '__main__':
    import numpy as np
    # print(np.average(count_score(candidates = ["This"," ", "is"," ", "a"," ", "cat"],
    #             reference = [["This"," ", "is"," ","not"," ", "a"," ", "cat"]])[0][:4]))
    print(np.average(count_score(candidates = ["This","is","a", "cat"],
                reference = [["This", "is","not","a", "cat"]])[0][:4]))
    print(np.average(count_score(candidates = "This is a cat",
                reference = ["This is not a cat"])[0][:4]))