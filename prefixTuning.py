import torch
from transformers import BartPretrainedModel, GPT2PreTrainedModel
from torch import nn


class PrefixTuningBart(BartPretrainedModel):

    use_cross_prefix = True  
    use_encoder_prefix = True

    def __init__(self, config, hparams, qkv_trans=False):
        super().__init__(config)   

        self.match_n_layer = config.decoder_layers  
        self.match_n_head = config.decoder_attention_heads  
        self.n_embd = config.d_model
        self.match_n_embd = self.n_embd // self.match_n_head   
        self.tuning_mode = hparams.tuning_mode

        self.mid_dim = hparams.mid_dim
        self.prefix_dropout = hparams.prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.preseqlen = hparams.preseqlen
        self.input_tokens = torch.arange(self.preseqlen).long()

        self.wte_decoder_prefix = nn.Embedding(self.preseqlen, self.n_embd)
        if self.use_cross_prefix:
            self.wte_cross_prefix = nn.Embedding(self.preseqlen, self.n_embd)
        if self.use_encoder_prefix:
            self.wte_encoder_prefix = nn.Embedding(self.preseqlen, self.n_embd)

        self.control_trans_decoder = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
        )
        self.control_trans_decoder_2 = nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd) if not qkv_trans else None

        if self.use_cross_prefix:

            self.control_trans_cross_prefix = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.mid_dim),
                nn.Tanh(),
            )
            self.control_trans_cross_prefix_2 = nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd) if not qkv_trans else None

        if self.use_encoder_prefix:

            self.control_trans_encoder_prefix = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.mid_dim),
                nn.Tanh(),
            )
            self.control_trans_encoder_prefix_2 = nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd) if not qkv_trans else None

        if qkv_trans:

            self.decoder_key_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.mid_dim)
            )

            self.decoder_val_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.mid_dim)
            )

            self.decoder_qry_trans = nn.Linear(self.mid_dim, self.mid_dim)   
            self.control_trans_decoder_2 = nn.Linear(self.n_embd * 2, self.match_n_layer * 2 * self.n_embd)
            self.control_trans_decoder_3 = nn.Linear(self.mid_dim, self.n_embd)
            self.mid_to_emb_trans_decoder = nn.Linear(self.mid_dim, self.n_embd)

            if self.use_cross_prefix:
                self.cross_key_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )
                self.cross_val_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )
                self.cross_qry_trans = nn.Linear(self.mid_dim, self.mid_dim)
                self.control_trans_cross_prefix_2 = nn.Linear(self.n_embd * 2, self.match_n_layer * 2 * self.n_embd)
                self.control_trans_cross_3 = nn.Linear(self.mid_dim, self.n_embd)
                self.mid_to_emb_trans_cross = nn.Linear(self.mid_dim, self.n_embd)

            if self.use_encoder_prefix:
                self.enc_key_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )
                self.enc_val_trans = nn.Sequential(
                    nn.Linear(self.n_embd, self.mid_dim),
                    nn.Tanh(),
                    nn.Linear(self.mid_dim, self.mid_dim)
                )
                self.enc_qry_trans = nn.Linear(self.mid_dim, self.mid_dim)
                self.control_trans_encoder_prefix_2 = nn.Linear(self.n_embd * 2, self.match_n_layer * 2 * self.n_embd)
                self.control_trans_encoder_3 = nn.Linear(self.mid_dim, self.n_embd)
                self.mid_to_emb_trans_enc = nn.Linear(self.mid_dim, self.n_embd)

    def _shape(self, tensor, seq_len, bsz, num_heads, head_dim):
        return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def get_attn_output(self, qry, key, val, drop_out=0.0, head_num=8):
        """
        qry.size() == [b, seq_len. mid_dim]
        """
        if head_num <= 1:
            qry = qry * (self.mid_dim ** -0.5)
            attn_weights = torch.bmm(qry, key.transpose(1, 2))   
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=drop_out, training=self.training)
            attn_output = torch.bmm(attn_probs, val)  
        else:
            bsz = qry.size(0)
            mid_dim = qry.size(-1)
            tgt_len = qry.size(1)
            assert mid_dim % head_num == 0
            head_dim = mid_dim // head_num
            proj_shape = (bsz * head_num, -1, head_dim)
            qry = qry * (self.mid_dim ** -0.5)
            qry = self._shape(qry, tgt_len, bsz, head_num, head_dim).view(*proj_shape)   
            key = key.view(*proj_shape)  
            val = val.view(*proj_shape)  
            attn_weights = torch.bmm(qry, key.transpose(1, 2))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=drop_out, training=self.training)
            attn_output = torch.bmm(attn_probs, val)  

            attn_output = attn_output.view(bsz, head_num, tgt_len, head_dim)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, mid_dim)  

        return attn_output

    def get_prompt(self, bsz, return_dict=False):

        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        temp_control = self.wte_decoder_prefix(input_tokens)  
        past_key_values_decoder_store = self.control_trans_decoder(temp_control)
        past_key_values_decoder = self.control_trans_decoder_2(past_key_values_decoder_store)

        if self.use_cross_prefix:
            temp_control_cross = self.wte_cross_prefix(input_tokens)
            past_key_values_cross_store = self.control_trans_cross_prefix(temp_control_cross)  
            past_key_values_cross = self.control_trans_cross_prefix_2(past_key_values_cross_store)
        if self.use_encoder_prefix:
            temp_control_enc = self.wte_encoder_prefix(input_tokens)
            past_key_values_enc_store = self.control_trans_encoder_prefix(temp_control_enc)  
            past_key_values_enc = self.control_trans_encoder_prefix_2(past_key_values_enc_store)

        bsz, seqlen, _ = past_key_values_decoder.shape
        
        
        past_key_values_decoder = past_key_values_decoder.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
        past_key_values_decoder = self.dropout(past_key_values_decoder)
        
        past_key_values_decoder = past_key_values_decoder.permute([2, 0, 3, 1, 4]).split(2)  

        """cross prefix"""

        if self.use_cross_prefix:
            bsz, seqlen, _ = past_key_values_cross.shape
            past_key_values_cross = past_key_values_cross.view(bsz, seqlen, self.match_n_layer * 2,
                                                     self.match_n_head, self.match_n_embd)
            past_key_values_cross = self.dropout(past_key_values_cross)
            past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)  

        if self.use_encoder_prefix:

            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                           self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        """
        shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
        shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
        """

        prefix_past_kv_list = []
        for i, key_val in enumerate(past_key_values_decoder):

            prefix_past_kv = dict()

            prefix_key_decoder = key_val[0].contiguous()
            prefix_val_decoder = key_val[1].contiguous()

            prefix_past_kv["decoder"] = {
                    "prefix_key": prefix_key_decoder,
                    "prefix_value": prefix_val_decoder,
            }

            if self.use_cross_prefix:
                key_val_cross = past_key_values_cross[i]
                prefix_key_cross = key_val_cross[0].contiguous()
                prefix_val_cross = key_val_cross[1].contiguous()

                prefix_past_kv["encoder_decoder"] = {
                    "prefix_key": prefix_key_cross,
                    "prefix_value": prefix_val_cross,
                }

            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                prefix_key_encoder = key_val_enc[0].contiguous()
                prefix_val_encoder = key_val_enc[1].contiguous()

                prefix_past_kv["encoder"] = {
                    "prefix_key": prefix_key_encoder,
                    "prefix_value": prefix_val_encoder,
                }

            prefix_past_kv_list.append(prefix_past_kv)

        prefix_key_padding_mask = torch.ones(bsz, seqlen).int().to(temp_control.device)

        if not return_dict:
            return prefix_past_kv_list, prefix_key_padding_mask
        else:
            return {
                "prefix_past_kv_list": prefix_past_kv_list,
                "prefix_key_padding_mask": prefix_key_padding_mask,
                "past_key_values_decoder_store": temp_control,
                "past_key_values_cross_store": temp_control_cross,
                "past_key_values_enc_store": temp_control_enc
            }

    def get_prompt_2(self, bsz, prompt_dict, plm_emb_weight):
        
        

        """  decoder prefix  """

        klg_prefix_past_kv_list = prompt_dict["prefix_past_kv_list"]
        klg_prefix_key_padding_mask = prompt_dict["prefix_key_padding_mask"]
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)  
        temp_control = self.wte_decoder_prefix(input_tokens)  
        past_key_values_decoder_store = self.control_trans_decoder(temp_control)

        past_key_values_decoder_query = self.decoder_qry_trans(past_key_values_decoder_store)
        past_key_values_decoder_key = self.decoder_key_trans(prompt_dict["past_key_values_decoder_store"])
        past_key_values_decoder_value = self.decoder_val_trans(prompt_dict["past_key_values_decoder_store"])

        decoder_attn_output = self.get_attn_output(
            qry=past_key_values_decoder_query,
            key=past_key_values_decoder_key,
            val=past_key_values_decoder_value,
        )

        decoder_attn_output = self.mid_to_emb_trans_decoder(decoder_attn_output)   
        plm_emb_weight = plm_emb_weight.t()[None, :, :].repeat(bsz, 1, 1)  
        decoder_attn_output = torch.bmm(decoder_attn_output, plm_emb_weight)  
        decoder_attn_output_bow_logits = torch.softmax(decoder_attn_output, dim=-1)
        decoder_attn_output = torch.bmm(plm_emb_weight, decoder_attn_output_bow_logits.transpose(1, 2))  
        decoder_attn_output = decoder_attn_output.transpose(1, 2)    

        decoder_out_aux_trans = self.control_trans_decoder_3(past_key_values_decoder_store)
        decoder_out_aux_trans = torch.cat([decoder_attn_output, decoder_out_aux_trans], dim=-1)
        past_key_values_decoder = self.control_trans_decoder_2(decoder_out_aux_trans)

        if self.use_cross_prefix:
            temp_control_cross = self.wte_cross_prefix(input_tokens)
            past_key_values_cross_store = self.control_trans_cross_prefix(temp_control_cross)  

            past_key_values_cross_query = self.cross_qry_trans(past_key_values_cross_store)
            past_key_values_cross_key = self.cross_key_trans(prompt_dict["past_key_values_cross_store"])
            past_key_values_cross_value = self.cross_val_trans(prompt_dict["past_key_values_cross_store"])

            cross_attn_output = self.get_attn_output(
                qry=past_key_values_cross_query,
                key=past_key_values_cross_key,
                val=past_key_values_cross_value,
            )

            cross_attn_output = self.mid_to_emb_trans_cross(cross_attn_output)  
            cross_attn_output = torch.bmm(cross_attn_output, plm_emb_weight)  
            cross_attn_output_bow_logits = torch.softmax(cross_attn_output, dim=-1)
            cross_attn_output = torch.bmm(plm_emb_weight, cross_attn_output_bow_logits.transpose(1, 2))  
            cross_attn_output = cross_attn_output.transpose(1, 2)  

            cross_out_aux_trans = self.control_trans_cross_3(past_key_values_cross_store)
            cross_out_aux_trans = torch.cat([cross_attn_output, cross_out_aux_trans], dim=-1)
            past_key_values_cross = self.control_trans_cross_prefix_2(cross_out_aux_trans)

        if self.use_encoder_prefix:

            temp_control_enc = self.wte_encoder_prefix(input_tokens)
            past_key_values_enc_store = self.control_trans_encoder_prefix(temp_control_enc)  

            past_key_values_enc_query = self.enc_qry_trans(past_key_values_enc_store)
            past_key_values_enc_key = self.enc_key_trans(prompt_dict["past_key_values_enc_store"])
            past_key_values_enc_value = self.enc_val_trans(prompt_dict["past_key_values_enc_store"])

            enc_attn_output = self.get_attn_output(
                qry=past_key_values_enc_query,
                key=past_key_values_enc_key,
                val=past_key_values_enc_value,
            )

            enc_attn_output = self.mid_to_emb_trans_enc(enc_attn_output)  
            enc_attn_output = torch.bmm(enc_attn_output, plm_emb_weight)  
            enc_attn_output_bow_logits = torch.softmax(enc_attn_output, dim=-1)
            enc_attn_output = torch.bmm(plm_emb_weight, enc_attn_output_bow_logits.transpose(1, 2))  
            enc_attn_output = enc_attn_output.transpose(1, 2)  

            enc_out_aux_trans = self.control_trans_encoder_3(past_key_values_enc_store)
            enc_out_aux_trans = torch.cat([enc_attn_output, enc_out_aux_trans], dim=-1)
            past_key_values_enc = self.control_trans_encoder_prefix_2(enc_out_aux_trans)

        bsz, seqlen, _ = past_key_values_decoder.shape
        
        
        past_key_values_decoder = past_key_values_decoder.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                               self.match_n_embd)
        past_key_values_decoder = self.dropout(past_key_values_decoder)
        
        past_key_values_decoder = past_key_values_decoder.permute([2, 0, 3, 1, 4]).split(2)  

        """cross prefix"""

        if self.use_cross_prefix:
            bsz, seqlen, _ = past_key_values_cross.shape
            past_key_values_cross = past_key_values_cross.view(bsz, seqlen, self.match_n_layer * 2,
                                                               self.match_n_head, self.match_n_embd)
            past_key_values_cross = self.dropout(past_key_values_cross)
            past_key_values_cross = past_key_values_cross.permute([2, 0, 3, 1, 4]).split(2)  

        if self.use_encoder_prefix:
            bsz_enc, seqlen, _ = past_key_values_enc.shape
            past_key_values_enc = past_key_values_enc.view(bsz_enc, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                           self.match_n_embd)
            past_key_values_enc = self.dropout(past_key_values_enc)
            past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        """
        shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
        shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
        """

        prefix_past_kv_list = []
        for i, key_val in enumerate(past_key_values_decoder):

            prefix_past_kv = dict()

            prefix_key_decoder = key_val[0].contiguous()
            prefix_val_decoder = key_val[1].contiguous()

            if klg_prefix_past_kv_list is not None:
                prefix_key_decoder = torch.cat(
                    [prefix_key_decoder, klg_prefix_past_kv_list[i]["decoder"]["prefix_key"]], dim=-2)
                prefix_val_decoder = torch.cat(
                    [prefix_val_decoder, klg_prefix_past_kv_list[i]["decoder"]["prefix_value"]], dim=-2)

            prefix_past_kv["decoder"] = {
                "prefix_key": prefix_key_decoder,
                "prefix_value": prefix_val_decoder,
            }

            if self.use_cross_prefix:
                key_val_cross = past_key_values_cross[i]
                prefix_key_cross = key_val_cross[0].contiguous()
                prefix_val_cross = key_val_cross[1].contiguous()

                if klg_prefix_past_kv_list is not None:
                    prefix_key_cross = torch.cat(
                        [prefix_key_cross, klg_prefix_past_kv_list[i]["encoder_decoder"]["prefix_key"]], dim=-2)
                    prefix_val_cross = torch.cat(
                        [prefix_val_cross, klg_prefix_past_kv_list[i]["encoder_decoder"]["prefix_value"]], dim=-2)

                prefix_past_kv["encoder_decoder"] = {
                    "prefix_key": prefix_key_cross,
                    "prefix_value": prefix_val_cross,
                }

            if self.use_encoder_prefix:
                key_val_enc = past_key_values_enc[i]
                prefix_key_encoder = key_val_enc[0].contiguous()
                prefix_val_encoder = key_val_enc[1].contiguous()

                if klg_prefix_past_kv_list is not None:
                    prefix_key_encoder = torch.cat(
                        [prefix_key_encoder, klg_prefix_past_kv_list[i]["encoder"]["prefix_key"]], dim=-2)
                    prefix_val_encoder = torch.cat(
                        [prefix_val_encoder, klg_prefix_past_kv_list[i]["encoder"]["prefix_value"]], dim=-2)

                prefix_past_kv["encoder"] = {
                    "prefix_key": prefix_key_encoder,
                    "prefix_value": prefix_val_encoder,
                }

            prefix_past_kv_list.append(prefix_past_kv)

        prefix_key_padding_mask = torch.ones(bsz, seqlen).int().to(temp_control.device)
        if klg_prefix_key_padding_mask is not None:
            prefix_key_padding_mask = torch.cat([prefix_key_padding_mask, klg_prefix_key_padding_mask], dim=1)

        merge_attn_output_bow_logits = torch.cat([decoder_attn_output_bow_logits, cross_attn_output_bow_logits, enc_attn_output_bow_logits], dim=1)

        return prefix_past_kv_list, prefix_key_padding_mask, merge_attn_output_bow_logits

    def forward(self, input_ids, model, _prefix_past_kv_list=None,
                _prefix_key_padding_mask=None, **kwargs):

        bsz = input_ids.shape[0]

        if _prefix_past_kv_list is None and _prefix_key_padding_mask is None:
            _prefix_past_kv_list, _prefix_key_padding_mask = self.get_prompt(bsz)

        output = model(
            input_ids=input_ids,
            pref_past_kv_list=_prefix_past_kv_list,
            pref_key_padding_mask=_prefix_key_padding_mask,
            **kwargs
        )
        return output


class PrefixTuningGPT2(GPT2PreTrainedModel):

    def __init__(self, config, hparams, qkv_trans=False):
        super().__init__(config)   

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.n_embd = config.n_embd
        self.match_n_embd = self.n_embd // self.match_n_head   
        self.tuning_mode = hparams.tuning_mode
        self.mid_dim = hparams.mid_dim
        self.prefix_dropout = hparams.prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)
        self.preseqlen = hparams.preseqlen
        self.input_tokens = torch.arange(self.preseqlen).long()
        self.wte_decoder_prefix = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_decoder = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.Tanh(),
        )
        self.control_trans_decoder_2 = nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.n_embd) if not qkv_trans else None

        if qkv_trans:

            self.decoder_key_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.mid_dim)
            )

            self.decoder_val_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.mid_dim)
            )

            self.decoder_qry_trans = nn.Linear(self.mid_dim, self.mid_dim)   
            self.control_trans_decoder_2 = nn.Linear(self.n_embd * 2, self.match_n_layer * 2 * self.n_embd)
            self.control_trans_decoder_3 = nn.Linear(self.mid_dim, self.n_embd)

            self.mid_to_emb_trans = nn.Linear(self.mid_dim, self.n_embd)

    def _shape(self, tensor, seq_len, bsz, num_heads, head_dim):
        return tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    def get_attn_output(self, qry, key, val, drop_out=0.0, head_num=8):
        """
        qry.size() == [b, seq_len. mid_dim]
        """
        if head_num <= 1:
            qry = qry * (self.mid_dim ** -0.5)
            attn_weights = torch.bmm(qry, key.transpose(1, 2))   
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=drop_out, training=self.training)
            attn_output = torch.bmm(attn_probs, val)  
        else:
            bsz = qry.size(0)
            mid_dim = qry.size(-1)
            tgt_len = qry.size(1)
            assert mid_dim % head_num == 0
            head_dim = mid_dim // head_num
            proj_shape = (bsz * head_num, -1, head_dim)
            qry = qry * (self.mid_dim ** -0.5)
            qry = self._shape(qry, tgt_len, bsz, head_num, head_dim).view(*proj_shape)   
            key = key.view(*proj_shape)  
            val = val.view(*proj_shape)  
            attn_weights = torch.bmm(qry, key.transpose(1, 2))
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_probs = nn.functional.dropout(attn_weights, p=drop_out, training=self.training)
            attn_output = torch.bmm(attn_probs, val)  

            attn_output = attn_output.view(bsz, head_num, tgt_len, head_dim)
            attn_output = attn_output.transpose(1, 2)
            attn_output = attn_output.reshape(bsz, tgt_len, mid_dim)  

        return attn_output

    def get_prompt(self, bsz, return_dict=False):

        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)  

        temp_control = self.wte_decoder_prefix(input_tokens)  
        past_key_values_decoder_store = self.control_trans_decoder(temp_control)
        past_key_values_decoder = self.control_trans_decoder_2(past_key_values_decoder_store)
        bsz, seqlen, _ = past_key_values_decoder.shape
        past_key_values_decoder = past_key_values_decoder.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
        past_key_values_decoder = self.dropout(past_key_values_decoder)
        past_key_values_decoder = past_key_values_decoder.permute([2, 0, 3, 1, 4]).split(2)  

        """
        shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
        shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.
        """

        prefix_past_kv_list = []
        for i, key_val in enumerate(past_key_values_decoder):

            prefix_past_kv = dict()
            prefix_key_decoder = key_val[0].contiguous()
            prefix_val_decoder = key_val[1].contiguous()

            prefix_past_kv["decoder"] = {
                    "prefix_key": prefix_key_decoder,
                    "prefix_value": prefix_val_decoder,
            }

            prefix_past_kv_list.append(prefix_past_kv)

        prefix_key_padding_mask = torch.ones(bsz, seqlen).int().to(temp_control.device)

        if not return_dict:
            return prefix_past_kv_list, prefix_key_padding_mask
        else:
            return {
                "prefix_past_kv_list": prefix_past_kv_list,
                "prefix_key_padding_mask": prefix_key_padding_mask,
                "past_key_values_decoder_store": temp_control,
            }

    def get_prompt_2(self, bsz, prompt_dict, plm_emb_weight):

        """  decoder prefix  """

        klg_prefix_past_kv_list = prompt_dict["prefix_past_kv_list"]
        klg_prefix_key_padding_mask = prompt_dict["prefix_key_padding_mask"]
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)  
        temp_control = self.wte_decoder_prefix(input_tokens)  
        past_key_values_decoder_store = self.control_trans_decoder(temp_control)
        past_key_values_decoder_query = self.decoder_qry_trans(past_key_values_decoder_store)
        past_key_values_decoder_key = self.decoder_key_trans(prompt_dict["past_key_values_decoder_store"])
        past_key_values_decoder_value = self.decoder_val_trans(prompt_dict["past_key_values_decoder_store"])
        decoder_attn_output = self.get_attn_output(
            qry=past_key_values_decoder_query,
            key=past_key_values_decoder_key,
            val=past_key_values_decoder_value,
        )   

        decoder_attn_output = self.mid_to_emb_trans(decoder_attn_output)   
        plm_emb_weight = plm_emb_weight.t()[None, :, :].repeat(bsz, 1, 1)  
        decoder_attn_output = torch.bmm(decoder_attn_output, plm_emb_weight)  
        decoder_attn_output_bow_logits = torch.softmax(decoder_attn_output, dim=-1)
        decoder_attn_output = torch.bmm(plm_emb_weight, decoder_attn_output_bow_logits.transpose(1, 2))  
        decoder_attn_output = decoder_attn_output.transpose(1, 2)    

        decoder_out_aux_trans = self.control_trans_decoder_3(past_key_values_decoder_store)
        decoder_out_aux_trans = torch.cat([decoder_attn_output, decoder_out_aux_trans], dim=-1)
        past_key_values_decoder = self.control_trans_decoder_2(decoder_out_aux_trans)

        bsz, seqlen, _ = past_key_values_decoder.shape

        past_key_values_decoder = past_key_values_decoder.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                                            self.match_n_embd)
        past_key_values_decoder = self.dropout(past_key_values_decoder)
        past_key_values_decoder = past_key_values_decoder.permute([2, 0, 3, 1, 4]).split(2)
        prefix_past_kv_list = []
        for i, key_val in enumerate(past_key_values_decoder):

            prefix_past_kv = dict()

            prefix_key_decoder = key_val[0].contiguous()
            prefix_val_decoder = key_val[1].contiguous()

            if klg_prefix_past_kv_list is not None:
                prefix_key_decoder = torch.cat(
                    [prefix_key_decoder, klg_prefix_past_kv_list[i]["decoder"]["prefix_key"]], dim=-2)
                prefix_val_decoder = torch.cat(
                    [prefix_val_decoder, klg_prefix_past_kv_list[i]["decoder"]["prefix_value"]], dim=-2)

            prefix_past_kv["decoder"] = {
                "prefix_key": prefix_key_decoder,
                "prefix_value": prefix_val_decoder,
            }

            prefix_past_kv_list.append(prefix_past_kv)

        prefix_key_padding_mask = torch.ones(bsz, seqlen).int().to(temp_control.device)
        if klg_prefix_key_padding_mask is not None:
            prefix_key_padding_mask = torch.cat([prefix_key_padding_mask, klg_prefix_key_padding_mask], dim=1)

        return prefix_past_kv_list, prefix_key_padding_mask, decoder_attn_output_bow_logits

    def forward(self, input_ids, model, _prefix_past_kv_list=None,
                _prefix_key_padding_mask=None, merge_attn_output_bow_logits=None, **kwargs):

        bsz = input_ids.shape[0]

        if _prefix_past_kv_list is None and _prefix_key_padding_mask is None:
            _prefix_past_kv_list, _prefix_key_padding_mask = self.get_prompt(bsz)

        output = model(
            input_ids=input_ids,
            pref_past_kv_list=_prefix_past_kv_list,
            pref_key_padding_mask=_prefix_key_padding_mask,
            merge_attn_output_bow_logits=merge_attn_output_bow_logits,
            **kwargs
        )

        return output
