import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)
    
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        # print(input.shape, recurrent.shape)
        
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        
        return output


class CRNN(nn.Module):
    
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        ks = [3, 3, 3, 3, 3, 3, 3, 3, 2]  # [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 1, 0]  # [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512, 512, 512]  # [64, 128, 256, 256, 512, 512, 512]
        
        cnn = nn.Sequential()
        
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                # cnn.add_module('layernorm{0}'.format(i), nn.GroupNorm(1, nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x32x?  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x16x?  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x8x?  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x?  # 512x2x16
        convRelu(6, True)
        convRelu(7)
        cnn.add_module('pooling{0}'.format(4),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x?  # 1024x2x16
        convRelu(8, True)  # 512x1x?  # 512x1x16
        # convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(512, nh, nh),
        #     BidirectionalLSTM(nh, nh, nclass))
        
        # self.load_state_dict(torch.load("./crnn.pth"))
        # for p in self.parameters():
        #     p.requires_grad = False
    
    def forward(self, input):
        # conv features
        output = self.cnn(input)
        # b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        output = output.squeeze(2)
        output = output.permute(0, 2, 1)  # [w, b, c]
        # # print(conv.shape)
        # # rnn features
        # output = self.rnn(conv)
        
        # print(output.shape)
        return output
