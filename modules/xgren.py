import torch
import torch.nn as nn
import numpy as np

from modules.text_extractor import MedCLIPTextModel
from modules.encoder_decoder import EncoderDecoder, subsequent_mask
from transformers import AutoTokenizer
from modules.visual_extractor import VisualExtractor_emebed


class XGRenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(XGRenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        self.visual_extractor = VisualExtractor_emebed(args)
        print('################################# using:', self.args.visual_extractor)

        # whether use kd (>0: use)
        if self.args.contras_loss_w > 0:
            print('################################# using kd, loss weight: ', self.args.contras_loss_w)
            self.text_encoder_kd = MedCLIPTextModel().cuda()

            if args.clip_pretrained_path is not None:
                self.load_from_medclip_text(args.clip_pretrained_path)

            if args.fix_text_encoder is True:
                for param in self.text_encoder_kd.parameters():
                    param.requires_grad = False
                print('################################# fix text encoder')
            else:
                print('################################# fine-tuning text encoder')

            self.reduce_head_im = nn.Linear(1024, 512, bias=False)  

            logit_scale_init_value=0.07
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/logit_scale_init_value)))


        # load topics
        self.use_topic = args.use_topic
        if self.use_topic:
            print('################################# using topics: ', args.topic_type)
            # self.topic_type = args.topic_type

            self.text_encoder_topics = MedCLIPTextModel().cuda()

            for param in self.text_encoder_topics.parameters():
                param.requires_grad = False

            self.topic_head = nn.Linear(768, self.args.d_vf, bias=False)

            topic_types_subs = {
                'iu': "cardiomegaly,scoliosis,fractures,effusion,thickening,pneumothorax,hernia,calcinosis,emphysema,pneumonia,edema,atelectasis,cicatrix,opacity,lesion,airspace disease,hypoinflation,medical device,normal,other",
                'knee': "knee,joint,fracture,alignment,effusion,lateral,soft,acute,tissue,evidence,degenerative,swelling,proximal,mild,satisfactory,patella,lucency,femoral,dislocation,anatomical,moderate,replacement,suprapatellar,changes,prosthesis,gas,complication,patellar,compartment,tissues,situ",
                'axr': "bowel,gas,obstruction,faecal,colon,loops,evidence,abdomen,rectum,supine,loading,pleural,lung,limits,tissue,contours,material,moderate,projection,cardiomediastinal,subdiaphragmatic,degenerative,consolidation,nonspecific,lumbar,quadrant,surgical,stomach",
                'hip': "hip,fracture,joint,femoral,degenerative,intact,pelvic,alignment,pelvis,bony,mild,periprosthetic,femur,acute,ring,pubic,proximal,bone,identified,sclerosis,bilateral,acetabular,fractures,enlocated,sacroiliac,moderate,lucency,symphysis",
                'wrist': "fracture,distal,alignment,radius,ulnar,radial,bony,styloid,dorsal,wrist,tissue,scaphoid,angulation,swelling,cast,carpal,acute,radio-carpal,lateral,mild,anatomical,degenerative,displacement,intact,plate,healing,interval,transverse",
                'shoulder': "joint,fracture,glenohumeral,humeral,alignment,head,shoulder,humerus,bony,degenerative,acromioclavicular,acute,clavicle,intact,space,dislocation,glenoid,fractures,identified,proximal,subacromial,lateral,tuberosity,change,cuff,inferior,calcification,visualised,tissue",
            }

            topic_types = 'abdomen, acetabular, acromioclavicular, acute, airspace disease, alignment, anatomical, angulation, atelectasis, bilateral, bone, bony, bowel, calcification, calcinosis, cardiomediastinal, cardiomegaly, carpal, cast, change, changes, cicatrix, clavicle, colon, compartment, complication, consolidation, contours, cuff, degenerative, dislocation, displacement, distal, dorsal, edema, effusion, emphysema, enlocated, evidence, faecal, femoral, femur, fracture, fractures, gas, glenohumeral, glenoid, head, healing, hernia, hip, humeral, humerus, hypoinflation, identified, inferior, intact, interval, joint, knee, lateral, lesion, limits, loading, loops, lucency, lumbar, lung, material, medical device, mild, moderate, nonspecific, normal, obstruction, opacity, other, patella, patellar, pelvic, pelvis, periprosthetic, plate, pleural, pneumonia, pneumothorax, projection, prosthesis, proximal, pubic, quadrant, radial, radio-carpal, radius, rectum, replacement, ring, sacroiliac, satisfactory, scaphoid, sclerosis, scoliosis, shoulder, situ, soft, space, stomach, styloid, subacromial, subdiaphragmatic, supine, suprapatellar, surgical, swelling, symphysis, thickening, tissue, tissues, transverse, tuberosity, ulnar, visualised, wrist'

            self.topics_ids = self.tokenizer(topic_types)

            self.topics_ids = torch.from_numpy(np.int32(np.array(self.topics_ids))).cuda()

            self.topics_masks_ = [1] * len(self.topics_ids)
            self.topics_masks = torch.from_numpy(np.array(self.topics_masks_)).cuda()
            # topics guided embeddings
            topics_embeds = self.text_encoder_topics(self.topics_ids.repeat(1,1), self.topics_masks.repeat(1,1), topics=True).detach()

            # merge tokens
            merge_id = [0]
            num_id = [1]
            for i in topic_types.split(', '):
                merge_id = np.concatenate([merge_id, self.tokenizer(i)[1:-2]], 0)
                num_id.append(len(merge_id))

            after_merge = []
            for m_i in range(len(num_id)-1):
                after_merge.append(topics_embeds[:, num_id[m_i]:num_id[m_i+1]].mean(1).unsqueeze(1))
            
            self.topics_embeds = torch.concat(after_merge, 1)

            # ##########################
            print('################################# using topic_splits')
            self.max_topic_length = 20
            self.topics_embeds_six = {}
            for topic_t in args.topic_type:
                topic_sub = topic_types_subs[topic_t].split(',')

                topics_embeds_sub = []
                for t_sub in topic_sub:
                    idx = topic_types.split(', ').index(t_sub)
                    topics_embeds_sub.append(self.topics_embeds[:,idx:idx+1,:])

                self.topics_embeds_six[topic_t] = torch.concat(topics_embeds_sub, 1)[:, 0:self.max_topic_length]
            self.topics_embeds = self.topics_embeds_six


    def load_from_medclip_text(self, checkpoint):
        state_dict = torch.load(checkpoint)
        new_state_dict = {}
        for key in state_dict.keys():
            if 'text_model' in key:
                new_state_dict[key.replace('text_model.','')] = state_dict[key]
        missing_keys, unexpected_keys = self.text_encoder_kd.load_state_dict(new_state_dict, strict=False)
        print('missing keys:', missing_keys)
        print('unexpected keys:', unexpected_keys)
        print('load model weight from:', checkpoint)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def compute_logits(self, img_emb, text_emb):
        self.logit_scale.data = torch.clamp(self.logit_scale.data, 0, 4.6052)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_emb, img_emb.t()) * logit_scale
        return logits_per_text

    def prepare_seq(self, seq=None):

        if seq is not None:
            # crop the last one
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return seq, seq_mask

    def forward(self, images, image_tags, targets=None, reports_ids_bert=None, reports_masks_bert=None, mode='train'):

        vis_feats_0, fc_feats_0, contras_im0 = self.visual_extractor(images[:, 0])
        vis_feats_1, fc_feats_1, contras_im1 = self.visual_extractor(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        vis_feats = torch.cat((vis_feats_0, vis_feats_1), dim=1)

        if self.args.contras_loss_w > 0:
            contras_im = torch.cat((contras_im0, contras_im1), dim=1)
            contras_im = self.reduce_head_im(contras_im)

        if self.use_topic:
            # match knowledge
            topics_embeds = [self.topics_embeds[key] for key in image_tags]
            topics_embeds = torch.concat(topics_embeds, 0)

            topics_embeds = self.topic_head(topics_embeds)
            
            # directly concat
            att_feats = torch.cat((vis_feats, topics_embeds), dim=1)

        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')

            if self.args.contras_loss_w > 0:
                seq_bert, seq_mask_bert = self.prepare_seq(reports_ids_bert)
                contras_text = self.text_encoder_kd(seq_bert, seq_mask_bert)

                if self.args.fix_text_encoder is True:
                    contras_text = contras_text.detach()
                logits_per_text = self.compute_logits(contras_im, contras_text)
            else:
                logits_per_text = None

        elif mode == 'sample':
            output, _,  = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            logits_per_text = None
        else:
            raise ValueError

        return output, logits_per_text
