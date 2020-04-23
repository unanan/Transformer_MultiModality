import logging
from util.bleu import Bleu         # validating metrics: bleu (referenced open codes on Github)

class MetricWrapper:
    " Validate Metrics wrapper. "
    
    def __init__(self, index2words, start_symbol, end_symbol, pad_symbol, metric=Bleu()):
        '''
        Args:
            index2words:  (dict) e.g.
            start_symbol: (str of int / int) should be the id of start symbol.
            end_symbol:   (str of int / int) should be the id of end symbol.
            pad_symbol:   (str of int / int) should be the id of pad symbol.
            metric:       (class) only support Bleu() now
        '''
        self.index2words = index2words
        self.metric = metric
        self.Ngram = metric.n
        
        if isinstance(start_symbol, str) and start_symbol.isnumeric():
            self.start_symbol = int(start_symbol)
        elif isinstance(start_symbol, int):
            self.start_symbol = start_symbol
        else:
            raise ValueError(f"Invalid start_symbol:{start_symbol}")
        
        if isinstance(end_symbol, str) and end_symbol.isnumeric():
            self.end_symbol = int(end_symbol)
        elif isinstance(end_symbol, int):
            self.end_symbol = end_symbol
        else:
            raise ValueError(f"Invalid end_symbol:{end_symbol}")

        if isinstance(pad_symbol, str) and pad_symbol.isnumeric():
            self.pad_symbol = int(pad_symbol)
        elif isinstance(pad_symbol, int):
            self.pad_symbol = pad_symbol
        else:
            raise ValueError(f"Invalid pad_symbol:{pad_symbol}")
            
    
    def intarr2str(self, intarr):
        '''
        Add logics of processing the int array of sequences. Cut off the <START>, <END>, <PAD>.
        
        Args:
            intarr: (ndarray) with int elements
        Returns:
            strlist: (nested lists of words) [["a","b","c"], ["g","h","j"], ]
        '''
        strlist = []

        for item in intarr:
            strlist_ = []
            for wordid in item:
                if wordid == self.start_symbol: #TODO
                    strlist_ = [self.index2words[str(self.start_symbol)]]
                    continue
                if wordid == self.end_symbol or wordid == self.pad_symbol:
                    break
                strlist_.append(self.index2words[str(wordid)])
            
            if len(strlist_)<1:
                strlist.append(None)
                logging.error(f"MetricWrapper: strlist_ is None.")
            else:
                strlist.append(strlist_)
            
        return strlist
    
    
    def __call__(self, out, trg):
        assert out.shape[0] == trg.shape[0], f"out:{out.shape[0]} trg:{trg.shape[0]}, dim-1 of out and trg must be the same."
        
        candidate_list = self.intarr2str(out.cpu().numpy())  # [["a","b","c"], ["g","h","j"], ]
        reference_list = self.intarr2str(trg.cpu().numpy())  # [["a","b","c"], ["g","h","j"], ]
        scores, _ = self.metric(candidates=candidate_list,
                                references=reference_list)  # scores: [[0.7, 0.6, 0.5, 0.2], [0.5, 0.5, 0.4, 0.1]]
        return scores