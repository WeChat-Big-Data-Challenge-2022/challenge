import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from category_id_map import CATEGORY_ID_LIST


# class MultiModal(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
#         self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
#                                  output_size=args.vlad_hidden_size, dropout=args.dropout)
#         self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
#         bert_output_size = 768
#         self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
#         self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
#         self.text_classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
#         self.vision_classifier = nn.Linear(args.vlad_hidden_size, len(CATEGORY_ID_LIST))
        
#         self.text_layernorm = nn.LayerNorm(bert_output_size)
#         self.vision_layernorm = nn.LayerNorm(args.vlad_hidden_size)
#         self.fuse_layernorm = nn.LayerNorm(args.vlad_hidden_size + bert_output_size)
#         self.dropout = nn.Dropout(args.dropout)
        

#     def forward(self, inputs, inference=False):
#         bert_embedding = self.bert(inputs['title_input'], inputs['title_mask'])['pooler_output']
#         bert_embedding_asr = self.bert(inputs['asr_input'], inputs['asr_mask'])['pooler_output']
#         bert_embedding = bert_embedding + bert_embedding_asr
        
#         vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
#         vision_embedding = self.enhance(vision_embedding)

#         final_embedding = self.fusion([vision_embedding, bert_embedding])
#         prediction = self.classifier(final_embedding)
        
#         # norm
#         bert_embedding = self.text_layernorm(bert_embedding)
#         vision_embedding = self.vision_layernorm(vision_embedding)
        
#         text_pred = self.text_classifier(bert_embedding)
#         vision_pred = self.vision_classifier(vision_embedding)
#         prediction = 0.5 * text_pred + 0.5* vision_pred
        
#         if inference:
#             return torch.argmax(prediction, dim=1)
#         else:
#             return self.cal_loss(prediction, inputs['label'])

#     @staticmethod
#     def cal_loss(prediction, label):
#         label = label.squeeze(dim=1)
#         loss = F.cross_entropy(prediction, label)
#         with torch.no_grad():
#             pred_label_id = torch.argmax(prediction, dim=1)
#             accuracy = (label == pred_label_id).float().sum() / label.shape[0]
#         return loss, accuracy, pred_label_id, label


# pretext task for pretrain
class ImageCaption(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.multi_modal = MultiModalModel(args)
        bert_output_size = 768
        self.decoder = DecoderRNN(input_size=bert_output_size+args.vlad_hidden_size, hidden_size=512, 
                                  vocab_size=args.vocab_size)
    def forward(self, inputs):
        """
        :inputs: {"frame":tensor, "title":tensor, "ocr":tensor, "asr":tensor} 
        """
        bert_embedding, vision_embedding = self.multi_modal(inputs, names=["asr", "ocr"])
        encoder_out = torch.cat([bert_embedding, vision_embedding], dim=-1)
        
        # preds shape = (batch_size, max_title_length, vacab_size)
        preds = self.decoder(encoder_out, inputs["title_input"].shape[1])
        # labels shape = (batch_size, max_title_length)
        labels = inputs["title_input"]
        return self.cal_loss(preds, labels)
    
    @staticmethod
    def cal_loss(preds, label):
        """
        :preds: shape = (batch_size, max_title_length, vacab_size)
        :label: shape = (batch_size, max_title_length)
        """
        # preds shape = (batch_size, vacab_size, max_title_length)
        prediction = torch.transpose(preds, 2, 1)
        loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = loss_fn(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(preds, dim=-1)   # shape = (batch_size, max_title_length)
            # print(f"pred_label_id shape: {pred_label_id.shape}, label shape: {label.shape}")
            accuracy = (label == pred_label_id).float().sum() / (label.shape[0] * label.shape[1])
        return loss, accuracy, pred_label_id, label
        
        
# for finetune 
class MultiModal(nn.Module):
    def __init__(self, args):
        super(MultiModal, self).__init__()
        self.multi_modal = MultiModalModel(args)
        bert_output_size = 768
        self.fusion = ConcatDenseSE(args.vlad_hidden_size + bert_output_size, args.fc_size, args.se_ratio, args.dropout)
        
        # 分类器
        self.text_classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
        self.title_classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
        self.asr_classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
        self.ocr_classifier = nn.Linear(bert_output_size, len(CATEGORY_ID_LIST))
        self.vision_classifier = nn.Linear(args.vlad_hidden_size, len(CATEGORY_ID_LIST))
        self.classifier = nn.Linear(args.fc_size, len(CATEGORY_ID_LIST))
        
        self.text_layernorm = nn.LayerNorm(bert_output_size)
        self.title_layernorm = nn.LayerNorm(bert_output_size)
        self.asr_layernorm = nn.LayerNorm(bert_output_size)
        self.ocr_layernorm = nn.LayerNorm(bert_output_size)
        self.vision_layernorm = nn.LayerNorm(args.vlad_hidden_size)
        self.fuse_layernorm = nn.LayerNorm(args.fc_size)
        
        self.class_weight = nn.Parameter(torch.rand(6).reshape(6, 1, 1))
        self.cosine = nn.CosineSimilarity(dim=1)
        
    def forward(self,  inputs, inference=False):
        """
        :inputs: {"frame":tensor, "title":tensor, "ocr":tensor, "asr":tensor, "label"} 
        """
        text_embedding, vision_embedding = self.multi_modal(inputs, names=["title", "asr", "ocr"])
        
        # fusing
        fusing_embedding = self.fusion([vision_embedding, text_embedding[-1]])
        
        # norm
        title_embedding = self.title_layernorm(text_embedding[0])
        asr_embedding = self.asr_layernorm(text_embedding[1])
        ocr_embedding = self.ocr_layernorm(text_embedding[2])
        bert_embedding = self.text_layernorm(text_embedding[-1])
        vision_embedding = self.vision_layernorm(vision_embedding)
        fusing_embedding = self.fuse_layernorm(fusing_embedding)
        
        # shape = (batch_size, n_class)
        title_pred = self.title_classifier(title_embedding)
        asr_pred = self.asr_classifier(asr_embedding)
        ocr_pred = self.ocr_classifier(ocr_embedding)
        text_pred = self.text_classifier(bert_embedding)
        vision_pred = self.vision_classifier(vision_embedding)
        fusing_pred = self.classifier(fusing_embedding)
        
        # pretdiction = fusing_pred + 0.5*vision_pred + 0.5*text_pred
        pred = torch.stack([title_pred, asr_pred, ocr_pred, text_pred, vision_pred, fusing_pred], dim=0)
        prediction = (pred * self.class_weight).sum(dim=0)    # (batch_size, n_class)
        
        if inference:
            return torch.argmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])
    
    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label        
        
        
class MultiModalModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                 output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        bert_output_size = 768
        self.projection = nn.Linear(bert_output_size * 3, bert_output_size)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs, names):
        """
        :inputs: {"frame":tensor, "title":tensor, "ocr":tensor, "asr":tensor} 
        :names: ["title", "asr" or "ocr"] 
        """
        text_embedding = []
        for name in names:
            emb = self.bert(inputs[f"{name}_input"], inputs[f"{name}_mask"])['pooler_output']
            text_embedding.append(emb)
        # bert_embedding = sum(text_embedding) / len(text_embedding)
        bert_embedding = self.projection(torch.cat(text_embedding, dim=-1))
        text_embedding.append(bert_embedding)
        bert_embedding = text_embedding
        
        vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
        vision_embedding = self.enhance(vision_embedding)
        
        return bert_embedding, vision_embedding
    
    

class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding

    
# Decoder
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.dropout = dropout

        self.dropout = nn.Dropout(p=self.dropout)
        self.decoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.init_h = nn.Linear(input_size, hidden_size)
        self.init_c = nn.Linear(input_size, hidden_size)
        self.f_beta = nn.Linear(input_size, hidden_size)   # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        :encoder_out: tensor, shape = (batch_size, text_size+vision_size)
        """
        h = self.init_h(encoder_out)  # shape = (batch_size, hidden_size)
        c = self.init_c(encoder_out)  # shape = (batch_size, hidden_size)
        h = torch.stack([h for i in range(self.num_layers)], dim=0)
        c = torch.stack([c for i in range(self.num_layers)], dim=0)
        return h, c

    def forward(self, encoder_out, max_title_length):
        """
        :encoder_out: tensor, shape = (batch_size, text_size+vision_size)
        :max_title_length: as the name.
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights
        """
        batch_size = encoder_out.size(0)
        
        # shape = (batch_size, max_title_length, dimention)
        decoder_input = torch.stack([x.repeat(max_title_length, 1) for x in encoder_out], dim=0)
        
        # shape = (num_layers, batch_size, hidden_size)
        h, c = self.init_hidden_state(encoder_out)
       
        # output shape = (batch_size, max_title_length, hidden_size)
        # h_n, c_n shape = (num_layers, batch_size, hidden_size)
        output, (h_n, c_n) = self.decoder(decoder_input, (h, c))
        
        # preds shape = (batch_size, max_title_length, vacab_size)
        preds = self.fc(output)
        
        return preds