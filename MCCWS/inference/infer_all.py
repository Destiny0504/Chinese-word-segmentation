import argparse
import os
import time

import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import MCCWS.util
from MCCWS.model import CWSmodel
from MCCWS.dataset import CWSDataset


def preprocess(data, tokenizer: transformers.PreTrainedTokenizerFast):
  # Transformers' tokenizer will output the tokenized string's index.
  # Size of the tokenized_str is [Batch size, Sequence length]
  tokenized_str = tokenizer(
    [x[0] for x in data], truncation=True, max_length=512, padding='longest', return_tensors='pt'
  )
  return tokenized_str, [x[1] for x in data], [x[2] for x in data]


def get_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=32, help='batch size', type=int)
  parser.add_argument('--gpu', default=0, help="which gpu do you want to use", type=int)
  parser.add_argument('--start_checkpoint', default=0, help="start checkpoint", type=int)
  parser.add_argument('--end_checkpoint', default=0, help="last checkpoint", type=int)
  parser.add_argument('--checkpoint_step', default=0, help="each step between checkpoint", type=int)
  parser.add_argument('--testset_token', action='append', default=[], help="Select the dataset")
  parser.add_argument('--exp_name', default=None, help="name your experiment", type=str)
  parser.add_argument('--criteria', default=10, help="How many criteria do you want to infer?", type=int)
  parser.add_argument('--model_name', default='bert-base-chinese', help="Select the model from hugging face", type=str)
  parser.add_argument('--dataset_token', default='[AS]', help="The criteria token put in front of each data.", type=str)

  return parser.parse_args()


def main(args):
  DEVICE = 'cuda:' + str(args.gpu)
  EXP_FILE = args.exp_name
  # DEVICE = 'cpu'
  print(DEVICE)
  MODEL_NAME = args.model_name
  
  tokenizer = MCCWS.util.load_tokenizer(MODEL_NAME)

  dict_of_testset = MCCWS.util.load_testset(args.testset_token)

  model = CWSmodel(
    model_name=args.model_name, total_token=len(tokenizer), drop=0.0, criteria=args.criteria
  )

  writer = SummaryWriter(f'./exp/result/{EXP_FILE}')

  for ckpt in range(args.start_checkpoint, args.end_checkpoint + 2, args.checkpoint_step):
    model = MCCWS.util.load_model(
      model=model,
      exp_name=EXP_FILE,
      checkpoint=ckpt,
    )
    model = model.to(DEVICE)
    model.eval()
    try:
      os.makedirs('./exp/result/' + EXP_FILE)
    except:
      pass
    # MCCWS.util.valid(model=model, tokenizer=tokenizer,exp_file=EXP_FILE,step=ckpt, device=DEVICE)
    for dataset_name, dataset_path in dict_of_testset.items():
      dataset = CWSDataset(
        {dataset_name: dataset_path},
        train_set=False,
        criterion_token=args.dataset_token,
      )
      data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: preprocess(data=x, tokenizer=tokenizer),
      )
      test_data = tqdm(data_loader)
      with torch.no_grad():
        f = open(f'./exp/result/{EXP_FILE}/{ckpt}_{dataset_name}_{args.dataset_token}.txt', 'w')
        for data, text, short_input in test_data:

          data = data.to(DEVICE)
          segmentation = model.inference(data)

          segmentation = segmentation.argmax(dim=-1)

          # Decoding.
          for i in range(len(segmentation)):
            tmp = []
            ans = ''
            space_counter = 0
            for j in range(len(text[i])):
              if text[i][j] == ' ':
                tmp.append(ans)
                ans = ''
                space_counter += 1
                continue
              if segmentation[i][j - space_counter] < 2:
                if segmentation[i][j - space_counter] == 0 and len(ans) > 0:
                  tmp.append(ans)  # This is need since model can wrongly produce sequences like "MB".
                  ans = ''
                ans += text[i][j]
              elif segmentation[i][j - space_counter] == 2:
                tmp.append(ans + text[i][j])
                ans = ''
              else:
                tmp.append(ans)  # This is need since model can wrongly produce sequences like "BMS".
                tmp.append(text[i][j])
                ans = ''
            if len(ans) > 0:
              tmp.append(ans)
            if short_input[i] and tmp[-1][-1] != '\n':
              tmp.append('\n')
            f.write(' '.join(tmp))
        f.close()

    model = model.to('cpu')
    F1_score, OOV_recall = MCCWS.util.score(
      exp_file=f'{EXP_FILE}', criteria=args.testset_token[0], step=ckpt, dataset_token=args.dataset_token
    )

    with open(f'./exp/result/{EXP_FILE}/F1_score.txt', 'a') as f:
      f.write(f'checkpoint : {ckpt} F1 : {F1_score} OOV_recall : {OOV_recall}\n')
    if args.dataset_token != '[UNC]':
      writer.add_scalar(f'{args.testset_token[0]}/F1_score', F1_score, ckpt)
      writer.add_scalar(f'{args.testset_token[0]}/OOV_recall', OOV_recall, ckpt)
      if ckpt == args.end_checkpoint:
        time.sleep(1)
    else:
      writer.add_scalar(f'{args.dataset_token}/{args.testset_token[0]}_F1_score', F1_score, ckpt)
      writer.add_scalar(f'{args.dataset_token}/{args.testset_token[0]}_OOV_recall', OOV_recall, ckpt)
      print(f'checkpoint : {ckpt} F1 : {F1_score} OOV_recall : {OOV_recall}\n')

if __name__ == "__main__":
  main(args=get_args())
