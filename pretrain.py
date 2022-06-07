import logging
import os
import time
import torch
import json

from config import parse_args
from data_helper import create_pretrain_dataloaders
from model import ImageCaption
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate, convert_to_gpu


def validate(model, val_dataloader, args):
    model.eval()
    losses = []
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = convert_to_gpu(batch, args.device)
            loss, _, pred_label_id, label = model(batch)
            predictions.append(pred_label_id.cpu())
            labels.append(label.cpu())
            losses.append(loss.cpu().numpy())
           
    loss = sum(losses) / len(losses)
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    accuracy = (predictions == labels).float().sum() / labels.shape[0] / labels.shape[1]
    # results = evaluate(predictions, labels)

    model.train()
    return loss, accuracy


def train_and_validate(args):
    # 1. load data
    train_dataloader, val_dataloader = create_pretrain_dataloaders(args)

    # 2. build model and optimizers
    model = ImageCaption(args)
    optimizer, scheduler = build_optimizer(args, model)
    
    if args.device:
        model = model.to(args.device)
    
    # 3. training
    step = 0
    best_loss = 100000
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            batch = convert_to_gpu(batch, args.device)
            model.train()
            
            loss, accuracy, _, _ = model(batch)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy: {accuracy:.3f}")
            
        # 4. validation
        loss, accuracy = validate(model, val_dataloader, args)
        logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, accuracy: {accuracy:.3f}")

        # 5. save checkpoint
        if loss < best_loss:
            best_loss = loss
            state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
            torch.save({'epoch': epoch, 'model_state_dict': state_dict},
                       f'{args.saved_pretrain_model_path}/model_epoch_{epoch}_loss_{loss:.3f}_acc{accuracy:.3f}.bin')


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.saved_pretrain_model_path, exist_ok=True)
    logging.info("Pretraining/evaluation parameters: %s", json.dumps(vars(args), indent=4))

    train_and_validate(args)


if __name__ == '__main__':
    main()
