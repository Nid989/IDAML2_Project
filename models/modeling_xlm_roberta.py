import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.checkpoint
from typing import Optional, Tuple, Union
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaPreTrainedModel,
    XLMRobertaModel,
)
from transformers.modeling_outputs import TokenClassifierOutput
from dataclasses import dataclass
from models.CRF import LC_CRF

@dataclass
class TokenCRFClassifierOutput(TokenClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_scores: torch.FloatTensor = None
    predictions: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class XLMRobertaCRFforTokenClassification(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"Pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf = LC_CRF(nb_labels=config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None, # new addition of mask 
        reduction: Optional[torch.Tensor] = "none", # new addition of reduction
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:

        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        prediction_scores = None
        predictions = None
        if labels is not None:
            # loss_fct = CrossEntropyLoss(weight=self.ce_weights)
            # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = self.crf(emissions=logits,
                            tags=labels,
                            mask=mask,
                            reduction=reduction)
        else:
            with torch.no_grad():
                prediction_scores, predictions = self.crf.decode(emissions=logits)
            
        if not return_dict:
            output = (logits,) + outputs[2:]
            output =  ((loss,) + output) if loss is not None else output # append loss if not none

        return TokenCRFClassifierOutput(
            loss=loss,
            logits=logits,
            predictions=predictions,
            prediction_scores=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        ) 

class XLMRobertaForTokenClassification(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"Pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, ce_weights: torch.Tensor=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.ce_weights = ce_weights

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.ce_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )