''' Wrapping Transformer Architecture based on Pytorch.
    (better on pytorch >1.2, if lower, may you have to hack by yourself.)
    Training on multi-modalities, only supporting GPU(s) for training '''
import os
import time
import numpy as np
import argparse
import logging
import torch
from torch.utils.data import DataLoader

from model.transformer import TransformerCNN            # transformer main architecture
from util.common import split_data, calc_metrics, calc_bleus, batch_greedy_decode, subsequent_mask
from util.loss import CritWrapper, CrossEntropyLoss, KLDivLoss
from util.optimizer import OptWrapper, AdamW, SGDW   #(refer to open codes on Github)
from util.translator import GreedyTranslator
from util.log import setlogger                         # logging utils which help to record training& validating phase

from dataset.formula1106norm import Formula1106, FormulaSampler, load_voc       # dataset of formula (written by UnaGuo)


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
    parser.add_argument('--display_interval', type=int, default=50,
                        help='interval of steps between two info entries')
    parser.add_argument('--device_ids', type=str, default='0',
                        help='device indexes of GPU(s) which are planned to use e.g."1,2" or "3"')
    parser.add_argument('--resume', type=str, default='',
                        help='The weights file to resume training (Only support *.pth)')
    
    # Pre-training Settings
    parser.add_argument('--pretrain', type=bool, default=False,  #TODO
                        help='pretrain or not')
    parser.add_argument('--pretrain_max_epoch', type=int, default=50,
                        help='pretrain epoch num')
    parser.add_argument('--pretrain_batch_size', type=int, default=16,
                        help='batch size when pre-training')
    
    # Training & Validating Settings
    parser.add_argument('--max_epoch', type=int, default=10000,
                        help='train epoch num')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size when training')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='interval of epoches between two validation')
    parser.add_argument('--val_translator', type=str, default="greedy",
                        help='["greedy","???","????"]')
    parser.add_argument('--val_metrics', type=str, default="bleu",
                        help='["bleu","???","????"]')
    
    # Optimizer & Loss Settings
    parser.add_argument('--opt_type', type=str, default="adamw",
                        help='["adam","adamw","sgd","sgdw","adagrad"]')
    parser.add_argument('--crit_type', type=str, default="crossentropy",
                        help='["crossentropy","kldivloss"]')
    parser.add_argument('--noamopt_factor', type=float, default=1,
                        help='')
    parser.add_argument('--noamopt_warmup', type=int, default=400,
                        help='')
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='')
    
    args = parser.parse_args()
    
    return args


def pretrain(model, max_epoch, samplers, dataloaders, optimizer, criterion, device_count, output_dir, display_interval):
    phase = "Pretrain"
    
    logging.info(f"Start {phase}..")
    epoch = 0
    while epoch < max_epoch:
        logging.info(f"========= Epoch {epoch} {phase} =========")
        model.train()
        batch_losses = []
        for it, data in enumerate(dataloaders['train']):
            torch.cuda.synchronize()
            start = time.time()

            src, trg, loss_mask, trg_ntokens = split_data(data)
            
            # Forward
            trg_y1 = trg[:, :-1]
            seq_len = trg_y1.shape[-1]
            trg_mask = subsequent_mask(seq_len).cuda()

            out = model(src=src, tgt=trg_y1, tgt_mask=trg_mask)
            
            # Loss Calculating
            trg_y2 = trg[:, 1:]
            loss = criterion(out, trg_y2, loss_mask, norm=samplers["train"].batch_size)
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            
            torch.cuda.synchronize()
            end = time.time()

            # Update the difficulty
            samplers["train"].update_online(batch_loss)

            # Print the batch info
            if (it+epoch*samplers['train'].batch_num) % display_interval==0:
                logging.info(f"{phase} [{it}/{samplers['train'].batch_num}]\t"
                             f"{phase}_loss = {batch_loss:.3f},\t"
                             f"curr_lr = {optimizer.rate():.6f},\t"
                             f"time/batch = {end - start:.3f}s")


        avg_batch_loss = np.mean(batch_losses)
        med_batch_loss = np.median(batch_losses)
        max_batch_loss = np.max(batch_losses)
        var_batch_loss = np.var(batch_losses)

        # criterion.step()
        logging.info(f"{phase} Loss: average:{avg_batch_loss:.6f}, "
                     f"variance:{var_batch_loss:.2f}, "
                     f"median:{med_batch_loss:.2f}, "
                     f"max:{max_batch_loss:.2f}")

        epoch += 1
    
    model_state_dic = model.module.state_dict() if device_count > 1 else model.state_dict()
    torch.save(model_state_dic, os.path.join(output_dir, f"{phase}_latest.pth"))
    
    return


def train(model, max_epoch, samplers, dataloaders, optimizer, criterion, translator, device_count, output_dir, val_interval, display_interval):
    phase = "Train"
    
    logging.info(f"Start {phase}..")
    epoch = 0
    while epoch < max_epoch:
        logging.info(f"========= Epoch {epoch} {phase} =========")
        model.train()
        batch_losses = []
        for it, data in enumerate(dataloaders['train']):
            torch.cuda.synchronize()
            start = time.time()

            src, trg, loss_mask, trg_ntokens = split_data(data)
            
            # Forward
            trg_y1 = trg[:, :-1]
            seq_len = trg_y1.shape[-1]
            trg_mask = subsequent_mask(seq_len).cuda()
            
            out = model(src=src, tgt=trg_y1, tgt_mask=trg_mask)
            
            # Loss Calculating
            trg_y2 = trg[:, 1:]
            loss = criterion(out, trg_y2, loss_mask, norm=trg_y2.shape[0])
            
            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            
            torch.cuda.synchronize()
            end = time.time()
            
            # Update the difficulty
            samplers["train"].update_online(batch_loss)
            
            # Print the batch info
            if (it + epoch * samplers['train'].batch_num) % display_interval == 0:
                logging.info(f"{phase} [{it}/{samplers['train'].batch_num}]\t"
                             f"{phase}_loss = {batch_loss:.4f},\t"
                             f"curr_lr = {optimizer.rate():.6f},\t"
                             f"time/batch = {end - start:.3f}s")
        
        avg_batch_loss = np.mean(batch_losses)
        med_batch_loss = np.median(batch_losses)
        max_batch_loss = np.max(batch_losses)
        var_batch_loss = np.var(batch_losses)
        
        # criterion.step()
        logging.info(f"{phase} Loss: average:{avg_batch_loss:.4f}, "
                     f"variance:{var_batch_loss:.4f}, "
                     f"median:{med_batch_loss:.4f}, "
                     f"max:{max_batch_loss:.4f}")
        
        # Validating phase
        if epoch % val_interval == 0:
            val_epoch(model, epoch, samplers, dataloaders, criterion, translator, device_count, output_dir)
        
        
        epoch += 1
    
    return
            

def val_epoch(model, epoch, samplers, dataloaders, criterion, translator, device_count, output_dir):
    ''' Called in "train" '''
    phase = "Validate"
    model.eval()

    bestavg_bleu_score = 0.0
    bleu_scores = []
    batch_losses = []

    torch.cuda.synchronize()
    start = time.time()
    for val_it, val_data in enumerate(dataloaders['val']):
        src, trg, loss_mask, trg_ntokens = split_data(val_data)
        
        # Forward
        with torch.no_grad():
            trg_y1 = trg[:, :-1]
            seq_len = trg_y1.shape[-1]
            trg_mask = subsequent_mask(seq_len).cuda()
    
            out = model(src=src, trg=trg_y1, tgt_mask=trg_mask)
            
            # Loss Calculating
            trg_y2 = trg[:, 1:]
            batch_loss = criterion(out, trg_y2, loss_mask, norm=trg_y2.shape[0]).item()
            batch_losses.append(batch_loss)

            # Bleu Calculating
            ys = translator(model, src) # N x seqLength
            bleu1, bleu2, bleu3, bleu4 = calc_bleus(trg, ys, index2words) #TODO
            bleu1, bleu2, bleu3, bleu4 = calc_metrics(bleu1),calc_metrics(bleu2),calc_metrics(bleu3),calc_metrics(bleu4),
            bleu_scores.append(bleu4)
            
            
            if val_it % 10 == 0: # val display interval 10
                logging.info(f"{phase} [{val_it}/{samplers['val'].batch_num}]\t"
                             f"{phase}_loss = {batch_loss:.4f},\t"
                             f"bleu-1 = {bleu1:.4f} bleu-2 = {bleu2:.4f} bleu-3 = {bleu3:.4f} bleu-4 = {bleu4:.4f}")

    avg_batch_loss, med_batch_loss, max_batch_loss, var_batch_loss = calc_metrics(batch_losses)
    avg_bleu_score, med_bleu_score, max_bleu_score, var_bleu_score = calc_metrics(bleu_scores)

    torch.cuda.synchronize()
    end = time.time()
    
    logging.info(f"Epoch-{epoch} {phase} Loss: avg:{avg_batch_loss:.4f}, "
                 f"var:{var_batch_loss:.4f}, "
                 f"median:{med_batch_loss:.4f}, "
                 f"max:{max_batch_loss:.4f} || "
                 f"Bleu-4: avg:{avg_bleu_score:.4f}, "
                 f"var:{var_bleu_score:.4f}"
                 f"median:{med_bleu_score:.4f}, "
                 f"max:{max_bleu_score:.4f} "
                 f"time/batch = {end - start:.3f}s")
    
    # Save model
    if avg_bleu_score > bestavg_bleu_score:
        bestavg_bleu_score = avg_bleu_score
        model_state_dic = model.module.state_dict() if device_count > 1 else model.state_dict()
        torch.save(model_state_dic, os.path.join(output_dir, f"epoch_{epoch}_bleu_{bestavg_bleu_score}.pth"))

    return
    
    
def main(args):
    # Global Settings
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_ids.strip()
    device_count = len(args.device_ids.strip().split(","))
    output_date = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(args.output_dir, output_date)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    setlogger(os.path.join(output_dir, "train.log"))
    
    # Vocabulary Settings
    words2index, index2words = load_voc(args.voc_path)  # Vocabulary dict map
    vocab_len = len(words2index.keys())
    start_symbol = words2index["<START>"]
    pad_symbol = words2index["<PAD>"]
    end_symbol = words2index["<END>"]
    
    # Data Settings
    datasets = {split: Formula1106(args.anno_path,
                                   words2index=words2index,
                                   image_root=args.image_root,
                                   split=split) for split in ['train', 'val', 'test']}
    samplers = {split: FormulaSampler(batch_size=args.batch_size, dataset=datasets[split], ) for split in ['train', 'val', 'test']}
    dataloaders = {split: DataLoader(datasets[split],
                                     # batch_size = args.batch_size, shuffle = True if split=='train' else False,
                                     batch_sampler=samplers[split],
                                     pin_memory=True,
                                     num_workers=4,
                                     ) for split in ['train', 'val', 'test']}
    db_maxlen = max([datasets[split].maxlength for split in ['train', 'val', 'test']])
    
    # Model Settings
    model_nonedp = TransformerCNN(trg_vocab=vocab_len).cuda()  # none data parallel
    logging.info(model_nonedp)
    d_model = model_nonedp.d_model
    if args.resume != '':
        model_nonedp.load_state_dict(torch.load(args.resume))
    if device_count > 1:
        model = torch.nn.DataParallel(model_nonedp)  # Not Supported
    else:
        model = model_nonedp
    
    # Optimizer
    if args.opt_type =="adam":
        from torch.optim import Adam
        optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                            Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
                            )
    elif args.opt_type == "adamw":
        optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                            AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
                            )
    elif args.opt_type == "sgd":
        from torch.optim import SGD
        optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                            SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                            )
    elif args.opt_type == "sgdw":
        optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                               SGDW(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
                            )
    elif args.opt_type == "adagrad":
        from torch.optim import Adagrad
        optimizer = OptWrapper(d_model, args.noamopt_factor, args.noamopt_warmup,
                            Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
                                 eps=1e-9, weight_decay=args.weight_decay)
                            )
    else:
        ### Add other custom optimizers here..
        raise ValueError(f"Invalid opt_type:{args.opt_type}")
    
    # Criterion(Loss)
    if args.crit_type == "crossentropy":
        criterion = CritWrapper(generator=model_nonedp.generator, crit=CrossEntropyLoss())
    elif args.crit_type == "kldivloss":
        criterion = CritWrapper(generator=model_nonedp.generator, crit=KLDivLoss())
    else:
        ### Add other custom criterions here..
        raise ValueError(f"Invalid crit_type:{args.crit_type}")
    
    # Translator
    if args.val_translator == "greedy":
        translator = GreedyTranslator(max_len = db_maxlen, start_symbol = start_symbol)
    else:
        ### Add other custom translators here..
        raise ValueError(f"Invalid val_translator:{args.val_translator}")
    
    # Validate Metrics
    if args.val_metrics  == "bleu":
        pass  #TODO
    else:
        ### Add other custom metrics here..
        raise ValueError(f"Invalid val_metrics:{args.val_metrics}")
    
    
    # Pre-training
    if args.pretrain:
        samplers['train'].batch_size = args.pretrain_batch_size
        pretrain(model, args.pretrain_max_epoch, samplers, dataloaders, optimizer, criterion,
                 device_count, output_dir, args.display_interval)

    
    # Training
    samplers['train'].batch_size = args.batch_size
    train(model, args.max_epoch, samplers, dataloaders, optimizer, criterion, translator,
             device_count, output_dir, args.val_interval, args.display_interval)


if __name__ == '__main__':
    args=parse_opt()
    main(args)