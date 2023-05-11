import MCCWS.preprocess
import MCCWS.util
import argparse
import re


def get_args():

  parser = argparse.ArgumentParser()

  parser.add_argument(
    '--original_test_gold_path',
    default='',
    help="Give the path of test gold file which you want to preprocess.",
    type=str
  )
  parser.add_argument(
    '--test_file', default='', help="Give the path where to store the preprocessed test file.", type=str
  )

  return parser.parse_args()


def main(args):

  numbers = re.compile(r'(((-|\+)?(\d+)([\.|·/∶:]\d+)?%?)+)')
  english_char = re.compile(r'[A-Za-z_.]+')

  with open(args.original_test_gold_path, 'r') as f:
    answer = f.readlines()

  with open(args.test_file, 'r') as f:
    predicted = f.readlines()

  tmp = list()

  for idx, data in enumerate(answer):

    sentence = ''
    for char in predicted[idx]:
      if char == '0':
        end = numbers.search(data).span()[1]
        sentence += data[numbers.search(data).span()[0] : numbers.search(data).span()[1]]
        data = data[end : ]
      elif char == 'a':
        end = english_char.search(data).span()[1]
        sentence += data[english_char.search(data).span()[0] : english_char.search(data).span()[1]]
        data = data[end : ]
      else:
        sentence += char
    tmp.append(sentence)

  with open(args.test_file, 'w') as f:
    for sentence in tmp:
      f.write(f"{' '.join(sentence.split())}\n")


if __name__ == "__main__":
  main(args=get_args())
