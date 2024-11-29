# from resnet_blocks import ResNetBottleneckBlock
from se_resnet_blocks import SE_ResNetBottleneckBlock as ResNetBottleneckBlock
from se_resnet_blocks import SE_ResNetBottleneckBlock_renorm
from batch_renormalization import BatchRenormalization2D

import torch
from torch import nn
from torch.nn import Module


class ResEncoderModel(Module):
    def __init__(self, start_channels=256, conv_blocks=(10, 10, 10)):
        super(ResEncoderModel, self).__init__()
        # Input is 3 x 64 x 64
        # prep -> 256 x 32 x 32
        self.conv_blocks = conv_blocks  # start_channelsx32x32 -> (start_channels*2)x16x16 -> (start_channels*2)x8x8
        self.num_blocks = len(self.conv_blocks)
        self.start_channels = start_channels
        self.prep = nn.Sequential(
            nn.Conv2d(  # N' = (N-K+2P)//S + 1
                in_channels=3,
                out_channels=self.start_channels,
                kernel_size=7,
                stride=1,  # Let's not reduce twice
                padding=3
            ),
            nn.BatchNorm2d(
                num_features = self.start_channels,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        # Output 256x 32 x 32
        current_channels = self.start_channels

        self.resnet_blocks = nn.ModuleList()

        for block_idx, conv_block_num in enumerate(self.conv_blocks):
            resnet_block = nn.Sequential()

            for conv_block_idx in range(conv_block_num):

                is_downsampling_block = False

                if block_idx > 0 and conv_block_idx == 0:
                    is_downsampling_block = True

                resnet_block.add_module(
                    f'conv_{conv_block_idx}',
                    ResNetBottleneckBlock(
                        in_channels_block = current_channels,
                        is_downsampling_block = is_downsampling_block
                    )
                )

                if is_downsampling_block:
                    current_channels *= 2

            self.resnet_blocks.append(resnet_block)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        x = self.prep.forward(x)  # B,3,H,W -> B,start_channels,H//2,W//2
        # print(f'shape after prep {x.shape}')

        for i in range(self.num_blocks):  # B,start_channels,H//2,W//2 -> B,start_channels,H//2,W//2 -> B,start_channels*2,H//4,W//4 -> B,start_channels*4,H//8,W//8
            x = self.resnet_blocks[i].forward(x)
            # print(f'shape after resnet_block {i} {x.shape}')

        # B,start_channels*4,H//8,W//8 -> B,start_channels*4
        # print(f'shape after resnet {x.shape}')
        x = self.avg_pool.forward(x)
        x = torch.squeeze(x, dim=3)
        x = torch.squeeze(x, dim=2)
        # print(f'shape after avg_pool {x.shape}')

        return x


class ContextPredictionModel(Module):

    def __init__(self, in_channels=1024):
        super(ContextPredictionModel, self).__init__()

        self.in_channels = in_channels

        # Input will be in_channelsx7x7

        # Two sets of convolutional context networks - one for vertical, one for horizontal agregation.

        # Prediction 3 steps ahead. So I will have 8 outputs.
        # [0,2:6] predict->[3,4,5:6],[1,3:6] predict->[4,5,6:6]
        # [4,6:6] predict->[3,2,1:6],[3,5:6] predict->[2,1,0:6]

        # [6:0,2] predict->[6:3,4,5],[6:1,3] predict->[6:4,5,6]
        # [6:4,6] predict->[6:3,2,1],[6:3,5] predict->[6:2,1,0]

        self.context_layers = 3
        self.context_conv = nn.Sequential()

        #
        for layer_idx in range(self.context_layers):
            self.context_conv.add_module(f'batch_norm_{layer_idx}',nn.BatchNorm2d(self.in_channels)),
            self.context_conv.add_module(f'relu_{layer_idx}',nn.ReLU())
            self.context_conv.add_module(
                f'conv2d_{layer_idx}',
                nn.Conv2d(
                    in_channels = self.in_channels,
                    out_channels = self.in_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        self.context_conv.add_module(
            'adaptive_avg_pool',
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        # Y direction predictions, X direction predictions

        self.prediction_weights = nn.ModuleList([nn.ModuleList() for i in range(4)])

        for direction in range(2):
            for prediction_steps in range(3):
                self.prediction_weights[direction].append(
                    nn.Linear(
                        in_features = self.in_channels,
                        out_features = self.in_channels,
                    )
                )

    def forward(self, x):
        # x: B,C,7,7
        z_patches_list = []
        z_patches_loc_list = []

        for y1 in range(5):
            for x1 in range(5):
                y2 = y1 + 2
                x2 = x1 + 2

                z_patches = x[:, :, y1:y2+1, x1:x2+1]  # 获取3*3的编码块
                z_patches_loc = (y1+1, x1+1)  # Store middle of the 3x3 square，即3*3编码块的中心索引

                z_patches_list.append(z_patches)
                z_patches_loc_list += [z_patches_loc] * len(z_patches)  # 对于batch中的每一个数据都对应一个编码块位置索引

        z_patches_tensor = torch.cat(z_patches_list, dim = 0)  # 沿着batch维度进行拼接，B*25,C,3,3

        # B*25,C,3,3 ->　B*25,C
        context_vectors = self.context_conv.forward(z_patches_tensor)  # B*25,C,3,3 ->　B*25,C,1,1
        context_vectors = context_vectors.squeeze(dim=3)
        context_vectors = context_vectors.squeeze(dim=2)

        context_vectors_for_yp = []
        context_loc_for_yp = []

        context_vectors_for_xp = []
        context_loc_for_xp = []

        for v_idx in range(len(context_vectors)):
            y3 = z_patches_loc_list[v_idx][0]
            x3 = z_patches_loc_list[v_idx][1]

            if y3 == 1 or y3 == 2:
                context_vectors_for_yp.append(context_vectors[v_idx:v_idx+1])
                context_loc_for_yp.append(z_patches_loc_list[v_idx])

            if x3 == 1 or x3 == 2:
                context_vectors_for_xp.append(context_vectors[v_idx:v_idx+1])
                context_loc_for_xp.append(z_patches_loc_list[v_idx])

        context_vect_tensor_for_yp = torch.cat(context_vectors_for_yp, dim=0)  # B*10,C
        context_loc_for_yp_t = torch.tensor(context_loc_for_yp)  # B*10

        context_vect_tensor_for_xp = torch.cat(context_vectors_for_xp, dim=0)  # B*10,C
        context_loc_for_xp_t = torch.tensor(context_loc_for_xp)  # B*10

        all_predictions = []
        all_loc = []

        # 根据编码块前两行预测编码块后三行
        for steps_y in range(3):
            # 预测的编码信息，B*10,C -> B*10,C
            predictions = self.prediction_weights[0][steps_y].forward(context_vect_tensor_for_yp)
            all_predictions.append(predictions)
            steps_add = torch.tensor([steps_y + 2, 0])
            # 预测的编码信息对应的位置索引
            all_loc.append(context_loc_for_yp_t + steps_add)

        # 根据编码块前两列预测编码块后三列
        for steps_x in range(3):
            # 预测的编码信息，B*10,C -> B*10,C
            predictions = self.prediction_weights[1][steps_x].forward(context_vect_tensor_for_xp)
            all_predictions.append(predictions)
            steps_add = torch.tensor([0, steps_x + 2])
            # 预测的编码信息对应的位置索引
            all_loc.append(context_loc_for_xp_t + steps_add)
        # ret: 6*B*10,C    6*B*10,2
        ret = torch.cat(all_predictions, dim = 0), torch.cat(all_loc, dim = 0)

        return ret


class ResClassificatorModel(Module):

    def __init__(self, in_channels, num_classes, hiddenChannels_=1024, numResBlocks_=16):
        super(ResClassificatorModel, self).__init__()
        # Input is [Bxinput_channelsx7x7] shaped tensor

        self.num_classes = num_classes
        self.num_res_blocks = numResBlocks_
        self.in_channels = in_channels
        self.channels = hiddenChannels_

        self.prep = nn.Sequential(
            BatchRenormalization2D(in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = self.channels,
                kernel_size = 1,
                stride = 1,
                padding = 0
            ),
        )

        self.res_blocks = nn.Sequential()
        for i in range(self.num_res_blocks-1):
            self.res_blocks.add_module(
                f'res_block_{i}',
                SE_ResNetBottleneckBlock_renorm(
                    in_channels_block = self.channels,
                )
            )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(
            in_features = self.channels,
            out_features = self.num_classes
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # B,in_channels,H,W -> B,hiddenChannels_,H,W
        x = self.prep.forward(x)
        # B,hiddenChannels_,H,W -> B,hiddenChannels_,H,W
        x = self.res_blocks.forward(x)
        # B,hiddenChannels_,H,W -> B,hiddenChannels_,1,1
        x = self.avg_pool.forward(x)
        # B,hiddenChannels_,1,1 -> B,hiddenChannels_
        x = x.squeeze(dim=3)
        x = x.squeeze(dim=2)
        # B,hiddenChannels_ -> B,num_classes
        x = self.linear.forward(x)
        x = self.softmax.forward(x)

        return x
