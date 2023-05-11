import argparse
import json
import os

import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import MCCWS.util
from MCCWS.dataset import CWSDataset
from MCCWS.model import CWSmodel


def preprocess(
    data, tokenizer: transformers.PreTrainedTokenizerFast, replace_rate: float
):

    # Transformers' tokenizer will output the tokenized string's index.
    # Size of the tokenized_str is [Batch size, Sequence length]
    tokenized_str = tokenizer(
        [x[0] for x in data],
        truncation=True,
        max_length=512,
        padding="longest",
        return_tensors="pt",
    )

    # This mask is for training criteria classification.
    # The criteria token will randomly masked with a given probability(replace_rate).
    mask = torch.rand(tokenized_str.input_ids.size()[0]) < replace_rate

    # 21132 is the id of [UNC]
    tokenized_str.input_ids[:, 1] = tokenized_str.input_ids[:, 1].masked_fill_(
        mask, 21132
    )

    # The reason we need to minus one is because of [CLS] token.
    return tokenized_str, torch.LongTensor(
        [x[1] + [4] * (len(tokenized_str.input_ids[0]) - len(x[1]) - 1) for x in data]
    )


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--replace_rate", default=0.1, help="", type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--lr", default=2e-5, help="learning rate", type=float)
    parser.add_argument("--epoch", default=40, help="epoch", type=int)
    parser.add_argument("--seed", default=2873, help="random seed", type=int)
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        help="This is the value of optimizer AdamW's weight decay",
        type=float,
    )
    parser.add_argument(
        "--smooth_factor",
        default=0.1,
        help="This is the label smoothing factor for loss function",
        type=float,
    )
    parser.add_argument(
        "--gpu", default=0, help="Select the model for training.", type=int
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Select the model from hugging face.",
        type=str,
    )
    parser.add_argument(
        "--exp_name", default="cws_model", help="Name your experiment.", type=str
    )
    parser.add_argument(
        "--trainset_token", action="append", default=[], help="Select the dataset"
    )
    parser.add_argument(
        "--start_log", default=50000, help="The step that start logging", type=int
    )
    parser.add_argument(
        "--accumulation_step", default=2, help="Gradient accumulation step", type=int
    )
    parser.add_argument("--save_step", default=2000, help="Saving intervals", type=int)

    return parser.parse_args()


def main(args):

    EPOCH = args.epoch
    DEVICE = f"cuda:{args.gpu}"
    # DEVICE = f'cpu'
    LOG_STEP = args.save_step
    SAVE_STEP = args.save_step
    MODEL_NAME = args.model_name
    EXP_FILE = args.exp_name

    assert MODEL_NAME is not None

    print(f"Using {DEVICE}")

    MCCWS.util.set_seed(args.seed)

    tokenizer = MCCWS.util.load_tokenizer(MODEL_NAME)

    dict_of_datasets = MCCWS.util.load_dataset(args.trainset_token)

    for name in dict_of_datasets.values():
        print(f"training on : {name}")

    dataset = CWSDataset(dict_of_datasets, train_set=True)

    try:
        os.makedirs(f"./exp/{EXP_FILE}")
        os.makedirs(f"./exp/log/{EXP_FILE}")
        with open(f"./exp/{EXP_FILE}/hyperparameters.json", "w") as f:
            f.write(json.dumps(args.__dict__))
    except:
        with open(f"./exp/{EXP_FILE}/hyperparameters.json", "w") as f:
            f.write(json.dumps(args.__dict__))

    count = 1

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: preprocess(
            data=x, tokenizer=tokenizer, replace_rate=args.replace_rate
        ),
    )

    model = CWSmodel(
        model_name=MODEL_NAME,
        total_token=len(tokenizer),
        drop=args.dropout,
        criteria=10,
    )

    model = model.to(DEVICE)
    model.train()
    print("In training mode" if model.training else "In testing mode")

    # declare loss_fn for training
    segmentation_loss_fn = nn.CrossEntropyLoss(
        ignore_index=4, label_smoothing=args.smooth_factor
    )
    criterion_loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(f"./exp/log/{EXP_FILE}")

    # declare optimizer
    optimizer = MCCWS.util.load_optimizer(
        lr=args.lr,
        model_param=model.named_parameters(),
        weight_decay=args.weight_decay,
    )

    # declare scheduler
    scheduler = MCCWS.util.load_scheduler(
        optimizer=optimizer,
        num_warmup_steps=(
            len(dataset) * EPOCH // args.accumulation_step // args.batch_size // 10
        )
        + 1,
        num_training_steps=(
            len(dataset) * EPOCH // args.accumulation_step // args.batch_size
        )
        + 1,
    )

    for epoch in range(EPOCH):
        train_data = tqdm(train_loader)
        loss = 0.0

        for data, target in train_data:
            target = target.to(DEVICE)
            data = data.to(DEVICE)

            # Pass the data through the whole model
            criteria, segmentation = model(data)

            criteria_target = target[:, 0]
            target = target[:, 1:]

            # Calculating the loss of criteria classification.
            criterion_loss = criterion_loss_fn(
                criteria.view(-1, 10), criteria_target.view(-1)
            )

            # tensor.view() needs the tensor be contagious. So we use reshape right here.
            # Calculating the loss of word segmentation.
            segmentation_loss = segmentation_loss_fn(
                segmentation.view(-1, 4),
                target.reshape(target.size()[0] * target.size()[1]),
            )

            # total loss = criteria classification loss + Chinese word segmentation loss
            loss = criterion_loss + segmentation_loss

            # Loss should be divided by args.accumulation_step, because every iter only contribute
            # loss / args.accumulation_step 's loss while we are stimulating the original
            # batch size's loss.
            loss = loss / args.accumulation_step

            # Calculate the gradient (default create_graph is False)
            loss.backward()

            # If the count % args.accumulation_step == 0, the we update the model.
            if count % args.accumulation_step == 0:

                # Update the model's parameter
                optimizer.step()

                # Refresh the gradient
                optimizer.zero_grad()
                scheduler.step()

            if count % LOG_STEP == 0:
                segmentation = torch.masked_select(
                    segmentation.argmax(dim=-1), (target < 4)
                )
                target = torch.masked_select(target, (target < 4))
                train_acc = (segmentation == target).sum()
                train_acc = train_acc.item() / (target < 4).sum().item()
                writer.add_scalar("training_acc", train_acc, count)
                writer.add_scalar("training_loss", loss, count)
                train_data.set_description(
                    f"Epoch :{epoch}" + f" loss :{round(loss.item(), 4)}"
                )

                # Save the model.
                if count % SAVE_STEP == 0 and count >= args.start_log:
                    with open(f"./exp/{EXP_FILE}/step_{count}.model", "wb") as f:
                        torch.save(model.state_dict(), f)
            count += 1
        # MCCWS.util.valid(model=model, tokenizer=tokenizer, exp_file=EXP_FILE, step=count, device=DEVICE)


if __name__ == "__main__":
    main(args=get_args())
