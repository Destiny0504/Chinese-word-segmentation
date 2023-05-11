import torch.nn as nn
import torch
import transformers


class CWSmodel(nn.Module):

  def __init__(
    self, model_name: str, total_token: int, drop: float = 0.0, criteria: int = 10):
    """
    ===================================================================================
      Usage :
                              Construct the Chinese word segmentor.

    ===================================================================================
      Input :
      
      total_token (int) :     Resize the mo

      one_type (bool) :       This argument is for changing the dataset into          \
                              traditional Chinese or simplified Chinese. 
      
    ===================================================================================
      Output :                
                              A dictionary that stored the path of dataset which is   \
                              going to use for training.

    ===================================================================================
    """

    super(CWSmodel, self).__init__()

    self.pretrained_model = transformers.BertModel.from_pretrained(model_name)
    self.pretrained_model.resize_token_embeddings(total_token)

    # Segmentor
    self.segmentor = nn.Linear(in_features=768, out_features=4)
    # self.segmentor = nn.Sequential(
    #   nn.Linear(in_features=768, out_features=48),
    #   nn.LayerNorm(48),
    #   nn.Dropout(p=0.1, inplace=False),
    #   nn.Linear(in_features=48, out_features=4)
    # )

    # Criteria classifier
    self.criteria = nn.Linear(in_features=768, out_features=criteria)
    self.dropout = nn.Dropout(p=drop, inplace=False)

  def forward(self, batch_x):
    """ Forward the CWS model

    Args:
        batch_x (_type_): _description_

    Returns:
        pooler_output (torch.Tensor) : The hidden representation of [CLS] token.  
        
        criteria (torch.Tensor) : The probabilty of each criteria.
        
        last_hidden_state (torch.Tensor): The probabilty of each character's {B, M, E, S} label.
    """

    # Forward the pretrained language model.
    model_predict = self.pretrained_model(**batch_x)

    # Criteria classification
    criteria = self.criteria(self.dropout(model_predict.last_hidden_state[:, 1, :]))

    # remove the [CLS] token
    model_predict.last_hidden_state = self.dropout(model_predict.last_hidden_state[:, 1:, :])

    # Word segmentation
    model_predict.last_hidden_state = self.segmentor(model_predict.last_hidden_state[:, 1:, :])

    
    return criteria, model_predict.last_hidden_state

  def inference(self, batch_x):
    """

    Args:
        batch_x (_type_): _description_

    Returns:
        pooler_output (torch.Tensor) : The hidden representation of [CLS] token.  
        
        last_hidden_state (torch.Tensor): The probabilty of each character's {B, M, E, S} label.
    """
    model_predict = self.pretrained_model(**batch_x)

    # the rest of the tokens
    model_predict.last_hidden_state = self.segmentor(model_predict.last_hidden_state[:, 2:, :])

    return model_predict.last_hidden_state