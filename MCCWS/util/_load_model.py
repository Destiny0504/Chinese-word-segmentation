import torch

import MCCWS.model


def load_model(model, exp_name, checkpoint: int = 0):
    """
  ===================================================================================
    Usage :
                            This function is for loading your model which have been \
                            trained.
  ===================================================================================
    Input :
    
    model (CWS) : The dataset's token are put in the list. In most cases, \
                            each token represented a dataset, but [ALL] will load   \
                            every dataset we have.

    one_type (bool) :       This argument is for changing the dataset into          \
                            traditional Chinese or simplified Chinese. 
    
  ===================================================================================
    Output :                
                            A dictionary that stored the path of dataset which is   \
                            going to use for training.

  ===================================================================================
  """
    model.load_state_dict(
        torch.load(
            f"./exp/{exp_name}/step_{checkpoint}.model",
            map_location="cpu",
        )
    )
    return model
