import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18
import random


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer' + str(i))(x)
            return x


class Midshare_module(nn.Module):
    def __init__(self, arch='resnet50', share_net_begin=1, share_net_end=2):
        super(Midshare_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net_begin
        self.share_net_end = share_net_end

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net >= 1:
                for i in range(self.share_net, self.share_net_end):
                    setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            if self.share_net >= 1:
                for i in range(self.share_net, self.share_net_end):
                    x = getattr(self.visible, 'layer' + str(i))(x)
            return x


# class integrating_net(nn.Module):
#     def __init__(self, arch='resnet50', share_net_begin=1,share_net_end=2):
#         super(integrating_net, self).__init__()
#
#         model_v = resnet50(pretrained=True,
#                            last_conv_stride=1, last_conv_dilation=1)
#         # avg pooling to global pooling
#         self.share_net = share_net_begin
#         self.share_net_end = share_net_end
#
#         if self.share_net == 0:
#             self.fused = nn.ModuleList()
#             self.fused.conv1 = model_v.conv1
#             self.fused.bn1 = model_v.bn1
#             self.fused.relu = model_v.relu
#             self.fused.maxpool = model_v.maxpool
#             if self.share_net >= 0:
#                 for i in range(1, self.share_net_end):
#                     setattr(self.fused, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))
#         else:
#             self.fused = nn.ModuleList()
#             self.fused.conv1 = model_v.conv1
#             self.fused.bn1 = model_v.bn1
#             self.fused.relu = model_v.relu
#             self.fused.maxpool = model_v.maxpool
#             if self.share_net >= 0:
#                 for i in range(1, self.share_net_end):
#                     setattr(self.fused, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))
#
#     def forward(self, x):
#         if self.share_net == 0:
#             x = self.fused.conv1(x)
#             x = self.fused.bn1(x)
#             x = self.fused.relu(x)
#             x = self.fused.maxpool(x)
#             for i in range(1, self.share_net_end):
#                 x = getattr(self.fused, 'layer' + str(i))(x)
#             return x
#         else:
#             if self.share_net >= 1:
#                 for i in range(1, self.share_net_end):
#                     x = getattr(self.fused, 'layer'+str(i))(x)
#             return x
class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.thermal, 'layer' + str(i), getattr(model_t, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.thermal, 'layer' + str(i))(x)
            return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base, 'layer' + str(i), getattr(model_base, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer' + str(i))(x)
            return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='off', gm_pool='on', arch='resnet50', share_net=1, share_net_end=2, pcb='on',
                 local_feat_dim=256, num_strips_first=1, num_strips_second=2):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        self.Midshare_resnet1 = Midshare_module(arch=arch, share_net_begin=share_net, share_net_end=share_net_end)
        self.Midshare_resnet2 = Midshare_module(arch=arch, share_net_begin=share_net, share_net_end=share_net_end)
        self.local1 = base_resnet(arch=arch, share_net=share_net_end)
        self.local2 = base_resnet(arch=arch, share_net=share_net_end)
        self.local3 = base_resnet(arch=arch, share_net=share_net_end)

        self.non_local = no_local
        self.pcb = pcb
        if self.non_local == 'on':
            pass

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool

        if self.pcb == 'on':
            self.num_stripes_fs = num_strips_first
            local_conv_out_channels = local_feat_dim

            self.local_conv_list_first = nn.ModuleList()
            for _ in range(self.num_stripes_fs):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_list_first.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))

            self.fc_list_first = nn.ModuleList()
            for _ in range(self.num_stripes_fs):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list_first.append(fc)
            ####################################################################
            self.num_stripes_sd = num_strips_second
            self.local_conv_second = nn.ModuleList()
            for _ in range(self.num_stripes_sd):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_second.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))

            self.fc_second = nn.ModuleList()
            for _ in range(self.num_stripes_sd):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_second.append(fc)
            ####################################################################

            self.local_conv_third = nn.ModuleList()
            for _ in range(6):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)
                self.local_conv_third.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))

            self.fc_third = nn.ModuleList()
            for _ in range(6):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_third.append(fc)

        else:
            self.bottleneck = nn.BatchNorm1d(pool_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.classifier = nn.Linear(pool_dim, class_num, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, modal=0):

        x1_avg = torch.mean(x1, dim=1, keepdim=True)
        x2_avg = torch.mean(x2, dim=1, keepdim=True)
        random_number1 = random.randint(0, 1)
        random_number2 = random.randint(0, 1)
        if modal == 0:
            x_vis_empty = x1
            x_the_empty = x2
            if random_number1 == 0:
                x_fused_vis = torch.cat((x1_avg, x1_avg, x1_avg), 1)
                x_fused_vis = self.visible_module(x_fused_vis)
                x_fused_the = self.thermal_module(x_the_empty)
                x_fused = torch.cat((x_fused_vis, x_fused_the), 0)
            elif random_number1 == 1:
                x_fused_the = torch.cat((x2_avg, x2_avg, x2_avg), 1)
                x_fused_vis = self.visible_module(x_vis_empty)
                x_fused_the = self.thermal_module(x_fused_the)
                x_fused = torch.cat((x_fused_vis, x_fused_the), 0)

            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

        elif modal == 1:
            x = self.visible_module(x1)
            x_fused_vis = torch.cat((x1_avg, x1_avg, x1_avg), 1)
            x_fused = self.visible_module(x_fused_vis)


        elif modal == 2:
            x = self.thermal_module(x2)
            x_fused_the = torch.cat((x2_avg, x2_avg, x2_avg), 1)
            x_fused = self.thermal_module(x_fused_the)

        # shared block
        if self.non_local == 'on':
            pass
        else:
            if random_number2 == 0:
                x1 = self.Midshare_resnet1(x)
                x2 = self.Midshare_resnet2(x_fused)
            else:
                x1 = self.Midshare_resnet1(x_fused)
                x2 = self.Midshare_resnet2(x)
            half_h = int(x1.size(2) / 2)

            x_local1 = torch.cat((x1[:, :, 0:1 * half_h, :], x2[:, :, 1 * half_h:2 * half_h, :]), dim=2)
            x_local2 = torch.cat((x2[:, :, 0:1 * half_h, :], x1[:, :, 1 * half_h:2 * half_h, :]), dim=2)
            x1 = x_local1
            x2 = x_local2

            x_local1 = self.local1(x1)
            x_local2 = self.local2(x2)
            half = int(x1.size(0) / 2)
            if modal == 0:
                if random_number1 == 0:
                    x_fused = torch.cat((x1[0: half, :, :, :], x2[half: 2 * half, :, :, :]), 0)
                elif random_number1 == 1:
                    x_fused = torch.cat((x2[0: half, :, :, :], x1[half: 2 * half, :, :, :]), 0)
            elif modal == 1:
                x_fused = x1
            elif modal == 2:
                x_fused = x2
            x_local3 = self.local3(x_fused)

        if self.pcb == 'on':
            feat1 = x_local1
            assert feat1.size(2) % self.num_stripes_fs == 0
            stripe_1 = int(feat1.size(2) / self.num_stripes_fs)
            local_feat_first = []
            logits_first = []
            for i in range(self.num_stripes_fs):
                if self.gm_pool == 'on':
                    pass
                else:
                    local_feat = F.max_pool2d(feat1[:, :, i * stripe_1: (i + 1) * stripe_1, :],
                                              (stripe_1, feat1.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list_first[i](local_feat.view(feat1.size(0), feat1.size(1), 1, 1))
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_first.append(local_feat)
                if hasattr(self, 'fc_list_first'):
                    logits_first.append(self.fc_list_first[i](local_feat))

            feat_all_first = [lf for lf in local_feat_first]
            feat_all_first = torch.cat(feat_all_first, dim=1)
            ##############################################################################
            feat1 = x_local2
            assert feat1.size(2) % self.num_stripes_fs == 0
            stripe = int(feat1.size(2) / self.num_stripes_fs)
            local_feat_second = []
            logits_second = []
            for i in range(self.num_stripes_fs):
                if self.gm_pool == 'on':
                    pass
                else:
                    local_feat = F.max_pool2d(feat1[:, :, i * stripe: (i + 1) * stripe, :],
                                              (stripe, feat1.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_second[i](local_feat.view(feat1.size(0), feat1.size(1), 1, 1))
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_second.append(local_feat)
                if hasattr(self, 'fc_second'):
                    logits_second.append(self.fc_second[i](local_feat))

            feat_all_second = [lf for lf in local_feat_second]
            feat_all_second = torch.cat(feat_all_second, dim=1)

            ##########################################
            #######################################################################################
            feat1 = x_local3
            assert feat1.size(2) % self.num_stripes_fs == 0
            stripe_1 = int(feat1.size(2) / 6)
            local_feat_third = []
            logits_third = []
            for i in range(6):
                if self.gm_pool == 'on':
                    pass
                else:
                    local_feat = F.max_pool2d(feat1[:, :, i * stripe_1: (i + 1) * stripe_1, :],
                                              (stripe_1, feat1.size(-1)))
                # shape [N, c, 1, 1]
                local_feat = self.local_conv_third[i](local_feat.view(feat1.size(0), feat1.size(1), 1, 1))
                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_third.append(local_feat)
                if hasattr(self, 'fc_third'):
                    logits_third.append(self.fc_third[i](local_feat))

            feat_all_third = [lf for lf in local_feat_third]
            feat_all_third = torch.cat(feat_all_third, dim=1)

            if self.training:
                return local_feat_first, logits_first, feat_all_first, local_feat_second, logits_second, feat_all_second, local_feat_third, logits_third, feat_all_third
            else:
                predict = torch.cat([feat_all_first, feat_all_second, feat_all_third], dim=1)
                return self.l2norm(predict)
        else:
            if self.gm_pool == 'on':
                b, c, h, w = x.shape
                x = x.view(b, c, -1)
                p = 3.0
                x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
            else:
                x_pool = self.avgpool(x)
                x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

            feat = self.bottleneck(x_pool)

            if self.training:
                return x_pool, self.classifier(feat)  # , scores
            else:
                return self.l2norm(x_pool), self.l2norm(feat)