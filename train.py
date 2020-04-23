''' Wrapping Transformer Architecture based on Pytorch.
    (better on pytorch >1.2, if lower, may you have to hack by yourself.)
    Training on multi-modalities, only supporting GPU(s) for training '''
import os
import time
import numpy as np
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import TransformerCNN            # transformer main architecture
from util.common import statistical_count, subsequent_mask, load_vocabulary
from util.loss import CrossEntropyLoss,LabelSmoothingLoss
from util.optimizer import OptWrapper, AdamW, SGDW      # referenced open codes on Github
from util.translator import GreedyTranslator
from util.metricwrapper import MetricWrapper
from util.bleu import Bleu
from util.log import setlogger                          # logging utils which help to record training& validating phase


device = torch.device("cuda")

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # Dataset Settings
    parser.add_argument('--anno_path', type=str,
                        default="",
                        help='annotation file with "json" format')
    parser.add_argument('--voc_path', type=str,
                        default="",
                        help='vocabulary file with "json" format')
    parser.add_argument('--image_root', type=str,
                        default="",
                        help="imgs' root folder, since it's provided the img names rather than img whole pathes in the anno file")
    
    # Weights Output Folder
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help='the root folder of output weights and logfile, which will be put under a sub-folder named'
                             ' with time. e.g."weights/20200411185133"')
    
    # Display Interval GPU Devices & Resume Checkpoints
    parser.add_argument('--device_ids', type=str, default='0,1,3',
                        help='device indexes of GPU(s) which are planned to use e.g."1,2"')
    parser.add_argument('--display_interval', type=int, default=50,
                        help='interval of steps between two info entries')
    parser.add_argument('--resume', type=str, default='',
                        help='The weights file to resume training (Only support *.pth)')
    
    # Pre-training Settings
    parser.add_argument('--pretrain', type=bool, default=False,  #TODO
                        help='pretrain or not')
    parser.add_argument('--pretrain_max_epoch', type=int, default=50,
                        help='pretrain epoch num')
    parser.add_argument('--pretrain_batch_size', type=int, default=12,
                        help='batch size when pre-training')
    
    # Training & Validating Settings
    parser.add_argument('--max_epoch', type=int, default=10000,
                        help='train epoch num')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch size when training')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='interval of epoches between two validation')
    parser.add_argument('--val_translator', type=str, default="greedy",
                        help='["greedy","???","????"]')
    parser.add_argument('--val_metrics', type=str, default="bleu",
                        help='["bleu","???","????"]')
    
    # Optimizer & Loss Settings
    parser.add_argument('--opt_type', type=str, default="adagrad",
                        help='["adam","adamw","sgd","sgdw","adagrad"]')
    parser.add_argument('--crit_type', type=str, default="kldivloss",
                        help='["crossentropy","kldivloss"]')
    parser.add_argument('--noamopt_factor', type=float, default=1,
                        help='')
    parser.add_argument('--noamopt_warmup', type=int, default=400,
                        help='')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='')
    
    args = parser.parse_args()
    
    return args


class Trainer:
    def __init__(self, args):
        # ------------------------- Initialization ---------------------------
        # region Global Settings
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids.strip()
        self.device_count = len(args.device_ids.strip().split(","))
        output_date = time.strftime("%Y%m%d%H%M%S")
        self.output_dir = os.path.join(args.output_dir, output_date)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        setlogger(os.path.join(self.output_dir, "train.log"))
        # endregion

        # region Vocabulary Settings
        self.words2index, self.index2words = load_vocabulary(args.voc_path)  # Vocabulary dict map
        self.vocab_len = len(self.words2index.keys())
        self.start_symbol = self.words2index["<START>"]
        self.pad_symbol = self.words2index["<PAD>"]
        self.end_symbol = self.words2index["<END>"]
        # endregion

        # region Model Settings
        model_nonedp = TransformerCNN(trg_vocab=self.vocab_len).to(device)  # none data parallel
        logging.info(model_nonedp)
        d_model = model_nonedp.d_model
        if args.resume != '':
            model_nonedp.load_state_dict(torch.load(args.resume))
        if self.device_count > 1:
            self.model = nn.DataParallel(model_nonedp)  # Not Supported
        else:
            self.model = model_nonedp
        # endregion
        
        # region Data Settings
        self.datasets = {split: Dataset(args.anno_path,
                                       words2index=self.words2index,
                                       image_root=args.image_root,
                                       split=split) for split in ['train', 'val', 'test']}
        self.samplers = {split: Sampler(batch_size=args.batch_size, dataset=self.datasets[split], ) for split in
                    ['train', 'val', 'test']}
        self.dataloaders = {split: DataLoader(self.datasets[split],
                                         # batch_size = args.batch_size, shuffle = True if split=='train' else False,
                                         batch_sampler=self.samplers[split],
                                         pin_memory=True,
                                         num_workers=4,
                                         ) for split in ['train', 'val', 'test']}
        db_maxlen = max([self.datasets[split].maxlength for split in ['train', 'val', 'test']])
        # endregion

        # region Optimizer Settings
        if args.opt_type == "adam":
            from torch.optim import Adam
            self.optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                                   Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
                                   )
        elif args.opt_type == "adamw":
            self.optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                                   AdamW(self.model.parameters(), lr=args.lr,
                                         betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
                                   )
        elif args.opt_type == "sgd":
            from torch.optim import SGD
            self.optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                                   SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                                   )
        elif args.opt_type == "sgdw":
            global optimizer
            self.optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                                   SGDW(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                                   )
        elif args.opt_type == "adagrad":
            from torch.optim import Adagrad
            self.optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                                   Adagrad(self.model.parameters(), lr=args.lr,
                                           eps=1e-9, weight_decay=args.weight_decay)
                                   )
        else:
            ### Add other custom optimizers here..
            raise ValueError(f"Invalid opt_type:{args.opt_type}")
        # endregion

        # region Criterion(Loss) Settings
        if args.crit_type == "crossentropy":
            self.criterion = CrossEntropyLoss(generator=model_nonedp.generator, ignore_index = self.pad_symbol)
        elif args.crit_type == "kldivloss":
            self.criterion = LabelSmoothingLoss(generator=model_nonedp.generator, ignore_index = self.pad_symbol)
        else:
            ### Add other custom criterions here..
            raise ValueError(f"Invalid crit_type:{args.crit_type}")
        # endregion

        # region Translator Settings
        if args.val_translator == "greedy":
            self.translator = GreedyTranslator(dataparallel=True if self.device_count > 1 else False,
                                          max_len=db_maxlen, start_symbol=self.start_symbol)
        else:
            ### Add other custom translators here..
            raise ValueError(f"Invalid val_translator:{args.val_translator}")
        # endregion

        # region Validate Metrics Settings
        if args.val_metrics == "bleu":
            self.metricwapper = MetricWrapper(self.index2words, self.start_symbol, self.end_symbol, self.pad_symbol, metric=Bleu())
        else:
            ### Add other custom metrics here..
            raise ValueError(f"Invalid val_metrics:{args.val_metrics}")
        # endregion
    
        # ---------------------------- Training -----------------------------
        # Pre-training: Design custom tasks in pre-training phase
        if args.pretrain:
            self.samplers['train'].batch_size = args.pretrain_batch_size
            self.pretrain(args.pretrain_max_epoch, args.display_interval)
    
        # Training
        self.samplers['train'].batch_size = args.batch_size
        self.train(args.max_epoch, args.val_interval, args.display_interval)


    def split_data(self, data, device = device):
        src, trg, loss_mask, trg_ntokens = data
        src, trg, loss_mask, trg_ntokens = src.to(device), trg.to(device), loss_mask.to(device), trg_ntokens
        return src, trg, loss_mask, trg_ntokens


    def pretrain(self, max_epoch, display_interval):
        phase = "Pretrain"
        
        logging.info(f"Start {phase}..")
        epoch = 0
        while epoch < max_epoch:
            logging.info(f"========= Epoch {epoch} {phase} =========")
            self.model.train()
            batch_losses = []
            for it, data in enumerate(self.dataloaders['train']):
                torch.cuda.synchronize()
                start = time.time()
    
                src, trg, loss_mask, trg_ntokens = self.split_data(data, device)
                
                # Forward
                trg_y1 = trg[:, :-1]
                seq_len = trg_y1.shape[-1]
                trg_mask = subsequent_mask(seq_len).repeat(trg_y1.shape[0], 1, 1).cuda()
                
                out = self.model(src=src, trg=trg_y1, trg_mask=trg_mask)
                
                # Loss Calculating
                trg_y2 = trg[:, 1:]
                loss_mask = loss_mask[:, 1:]
                loss = self.criterion(out, trg_y2, loss_mask, norm=self.samplers["train"].batch_size)
                
                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                
                torch.cuda.synchronize()
                end = time.time()
    
                # Update the difficulty
                self.samplers["train"].update_online(batch_loss)
    
                # Print the batch info
                if (it+epoch*self.samplers['train'].batch_num) % display_interval==0:
                    logging.info(f"{phase} [{it}/{self.samplers['train'].batch_num}]\t"
                                 f"{phase}_loss = {batch_loss:.3f},\t"
                                 f"curr_lr = {self.optimizer.rate():.6f},\t"
                                 f"time/batch = {end - start:.3f}s")
    
            avg_batch_loss, med_batch_loss, max_batch_loss, var_batch_loss = statistical_count(batch_losses)
    
            logging.info(f"{phase} Loss: average:{avg_batch_loss:.6f}, "
                         f"variance:{var_batch_loss:.2f}, "
                         f"median:{med_batch_loss:.2f}, "
                         f"max:{max_batch_loss:.2f}")
    
            epoch += 1
        
        model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
        torch.save(model_state_dic, os.path.join(self.output_dir, f"{phase}_latest.pth"))
        
        return


    def train(self, max_epoch, val_interval, display_interval):
        phase = "Train"
        
        logging.info(f"Start {phase}..")
        epoch = 0
        while epoch < max_epoch:
            logging.info(f"========= Epoch {epoch} {phase} =========")
            self.model.train()
            batch_losses = []
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            for it, data in enumerate(self.dataloaders['train']):
                torch.cuda.synchronize()
                start = time.time()
    
                src, trg, loss_mask, trg_ntokens = self.split_data(data)
                
                # Forward
                trg_y1 = trg[:, :-1]
                seq_len = trg_y1.shape[-1]
                trg_mask = subsequent_mask(seq_len).repeat(trg_y1.shape[0],1,1).cuda()
                
                out = self.model(src=src, trg=trg_y1, trg_mask=trg_mask)
                
                # Loss Calculating
                trg_y2 = trg[:, 1:]
                loss_mask = loss_mask[:, 1:]
                loss = self.criterion(out, trg_y2, loss_mask, norm=1) #trg_y2.shape[0]
                
                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                torch.cuda.synchronize()
                end = time.time()
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                
                # Update the difficulty
                self.samplers["train"].update_online(batch_loss)
                
                # Print the batch info
                if (it + epoch * self.samplers['train'].batch_num) % display_interval == 0:
                    logging.info(f"{phase} [{it}/{self.samplers['train'].batch_num}]\t"
                                 f"{phase}_loss = {batch_loss:.4f},\t"
                                 f"curr_lr = {self.optimizer.rate():.6f},\t"
                                 f"time/batch = {end - start:.3f}s")
    
            avg_batch_loss, med_batch_loss, max_batch_loss, var_batch_loss = statistical_count(batch_losses)
            
            
            logging.info(f"{phase} Loss: average:{avg_batch_loss:.4f}, "
                         f"variance:{var_batch_loss:.4f}, "
                         f"median:{med_batch_loss:.4f}, "
                         f"max:{max_batch_loss:.4f}")
            

            # Validating phase
            if epoch % val_interval == 1:
                self.val_epoch(epoch)
            
            
            epoch += 1
        
        return
            

    def val_epoch(self, epoch):
        ''' Called in "train" '''
        phase = "Validate"
        self.model.eval()
    
        best_avg_score = 0.0
        scores = []
        losses = []
    
        torch.cuda.synchronize()
        start = time.time()
        for val_it, val_data in enumerate(self.dataloaders['val']):
            src, trg, loss_mask, trg_ntokens = self.split_data(val_data, device)
            
            # Forward
            with torch.no_grad():
                trg_y1 = trg[:, :-1]
                seq_len = trg_y1.shape[-1]
                trg_mask = subsequent_mask(seq_len).repeat(trg_y1.shape[0], 1, 1).cuda()
                
                out = self.model(src=src, trg=trg_y1, trg_mask=trg_mask)
                
                # Loss Calculating
                trg_y2 = trg[:, 1:]
                loss_mask = loss_mask[:, 1:]
                batch_loss = self.criterion(out, trg_y2, loss_mask, norm=1).item() #trg_y2.shape[0]
                losses.append(batch_loss)
    
                # Validate Metrics: Bleu Calculating
                ys = self.translator(self.model, src) # N x seqLength
                batch_scores = self.metricwapper(ys, trg)  #[[0.7, 0.6, 0.5, 0.2], [0.5, 0.5, 0.4, 0.1], ]  N x 4
                scores.extend(batch_scores)
                avg_score_, med_score_, max_score_, var_score_ = statistical_count(batch_scores)
                batchavg_bleu1,batchavg_bleu2,batchavg_bleu3,batchavg_bleu4 = avg_score_[:4]  #TODO
    
                if val_it % 10 == 0: # val display interval 10
                    logging.info(f"{phase} [{val_it}/{self.samplers['val'].batch_num}]\t"
                                 f"{phase}_loss = {batch_loss:.4f},\t"
                                 f"bleu-1 = {batchavg_bleu1:.4f} || bleu-2 = {batchavg_bleu2:.4f} || "
                                 f"bleu-3 = {batchavg_bleu3:.4f} || bleu-4 = {batchavg_bleu4:.4f}") #TODO
                    
    
        avg_loss, med_loss, max_loss, var_loss = statistical_count(losses)
        avg_score, med_score, max_score, var_score = statistical_count(scores)
    
        torch.cuda.synchronize()
        end = time.time()
        
        logging.info(f"Epoch-{epoch} {phase} Loss: avg:{avg_loss:.4f}, "
                     f"var:{var_loss:.4f}, "
                     f"median:{med_loss:.4f}, "
                     f"max:{max_loss:.4f} || "
                     f"Bleu-4: avg:{avg_score[3]:.4f}, "  #TODO
                     f"var:{var_score[3]:.4f}, "
                     f"median:{med_score[3]:.4f}, "
                     f"max:{max_score[3]:.4f} "
                     f"time/batch = {end - start:.3f}s")
        
        # Save model
        if avg_score[3] > best_avg_score:
            best_avg_score = avg_score[3]  #Bleu-4
            model_state_dic = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()
            torch.save(model_state_dic, os.path.join(self.output_dir, f"epoch_{epoch}_bleu_{best_avg_score}.pth"))
    
        return


if __name__ == '__main__':
    args=parse_opt()
    Trainer(args)