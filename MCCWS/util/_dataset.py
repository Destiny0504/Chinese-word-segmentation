from pathlib import Path


def load_dataset(trainset_token: list) -> dict:
    """
  ===================================================================================
    Usage :
                            This function is for loading the dataset which you are  \
                            going to train on.
  ===================================================================================
    Input :
    
    trainset_token (list) : The dataset's token are put in the list. In most cases, \
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
    for token in trainset_token:
        assert token in [
            "[AS]",
            "[CIT]",
            "[MSR]",
            "[PKU]",
            "[ALL]",
            "[CTB6]",
            "[CNC]",
            "[UD]",
            "[WTB]",
            "[SXU]",
            "[ZX]",
        ]

    return_dict = {}

    if "[AS]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[AS]"] = ["./data/trainset/as_train.txt"]

    if "[CIT]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[CIT]"] = ["./data/trainset/cityu_train.txt"]

    if "[MSR]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[MSR]"] = ["./data/trainset/msr_train.txt"]

    if "[PKU]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[PKU]"] = ["./data/trainset/pku_train.txt"]

    if "[CNC]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[CNC]"] = ["./data/trainset/cnc_train.txt"]

    if "[CTB6]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[CTB6]"] = ["./data/trainset/ctb6_train.txt"]

    if "[SXU]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[SXU]"] = ["./data/trainset/sxu_train.txt"]

    if "[UD]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[UD]"] = ["./data/trainset/ud_train.txt"]

    if "[WTB]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[WTB]"] = ["./data/trainset/wtb_train.txt"]

    if "[ZX]" in trainset_token or "[ALL]" in trainset_token:
        return_dict["[ZX]"] = ["./data/trainset/zx_train.txt"]

    return return_dict


def load_testset(testset_token: list) -> dict:
    """
===================================================================================
  Usage :
                          This function is for loading the dataset which you are  \
                          going to test on.
===================================================================================
  Input :
  
  testset_token (list) :  The dataset's token are put in the list. In most cases, \
                          each token represented a dataset, but [ALL] will load   \
                          every dataset we have.

  one_type (bool) :       This argument is for changing the dataset into          \
                          traditional Chinese or simplified Chinese. 
  
===================================================================================
  Output :                
                          A dictionary that stored the path of dataset which is   \
                          going to use for testing.

===================================================================================
  """
    for token in testset_token:

        assert token in [
            "[AS]",
            "[CIT]",
            "[MSR]",
            "[PKU]",
            "[ALL]",
            "[CTB6]",
            "[SXU]",
            "[UD]",
            "[CNC]",
            "[ZX]",
            "[WTB]",
            "[CTB6]",
        ]

    return_dict = {}

    if "[MSR]" in testset_token or "[ALL]" in testset_token:
        return_dict["[MSR]"] = ["./data/testset/msr_test.txt"]

    if "[PKU]" in testset_token or "[ALL]" in testset_token:
        return_dict["[PKU]"] = ["./data/testset/pku_test.txt"]

    if "[AS]" in testset_token or "[ALL]" in testset_token:
        return_dict["[AS]"] = ["./data/testset/as_test.txt"]

    if "[CIT]" in testset_token or "[ALL]" in testset_token:
        return_dict["[CIT]"] = ["./data/testset/cityu_test.txt"]

    if "[CTB6]" in testset_token or "[ALL]" in testset_token:
        return_dict["[CTB6]"] = ["./data/testset/ctb6_test.txt"]

    if "[CNC]" in testset_token or "[ALL]" in testset_token:
        return_dict["[CNC]"] = ["./data/testset/cnc_test.txt"]

    if "[SXU]" in testset_token or "[ALL]" in testset_token:
        return_dict["[SXU]"] = ["./data/testset/sxu_test.txt"]

    if "[UD]" in testset_token or "[ALL]" in testset_token:
        return_dict["[UD]"] = ["./data/testset/ud_test.txt"]

    if "[WTB]" in testset_token or "[ALL]" in testset_token:
        return_dict["[WTB]"] = ["./data/testset/wtb_test.txt"]

    if "[ZX]" in testset_token or "[ALL]" in testset_token:
        return_dict["[ZX]"] = ["./data/testset/zx_test.txt"]

    if "[CTB6_T]" in testset_token or "[ALL]" in testset_token:
        return_dict["[CTB6]"] = ["./data/testset/ctb6_test.txt"]

    return return_dict
