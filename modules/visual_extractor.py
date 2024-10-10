import torch
import torch.nn as nn
import torchvision.models as models
from modules.vits import create_vit

class VisualExtractor_emebed(nn.Module):
    def __init__(self, args):
        super(VisualExtractor_emebed, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        if 'vit' in self.visual_extractor:
            vit_grad_ckpt = False
            vit_ckpt_layer = 0
            image_size = 224

            vit_name = self.visual_extractor[4:]
            self.model, vision_width = create_vit(
                vit_name, image_size, vit_grad_ckpt, vit_ckpt_layer, 0)

            self.feature_dim = vision_width

            if self.pretrained:
                print('################ loading pretrained weights for vit')
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(msg)

            self.projection_head = nn.Linear(768, 512, bias=False)

        else:
            if self.pretrained:            
                print('################ loading pretrained weights for resnet')
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            # self.dropout = torch.nn.Dropout(0.5)
            # print('################ dropout:', self.dropout)
            self.projection_head = nn.Linear(2048, 512, bias=False)

    def forward(self, images):
        if 'vit' in self.visual_extractor:
            img_feat = self.model(images)
            img_embeds = self.projection_head(img_feat[:, 0].contiguous())
            return img_feat[:, 1:].contiguous(), img_feat[:, 0].contiguous(), img_embeds

        else:
            patch_feats = self.model(images)
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            img_embeds = self.projection_head(avg_feats)
            return patch_feats, avg_feats, img_embeds
