from dataset import display_trainable_params, set_seed, evaluate_modify
import torch
import argparse
import torch.optim as optim
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset import TrainDataset, collate_fn
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf

from models import CLAP_Dual



def parse_args():
    parser = argparse.ArgumentParser(description="training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()

    return args.config




def main():
    args = OmegaConf.load(parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(args.logging_dir)

    model = CLAP_Dual(layers=args.model_settings.layers, device=device)
    trainable_params, frozenable_params = display_trainable_params(model)

    # setting seed
    set_seed(args.seed)

    # load dataset
    print('loading dataset...')
    full_train_dataset = TrainDataset(csv_paths=args.train_csv_path)
    if args.val_size > len(full_train_dataset):
        raise ValueError("Validation set size cannot be larger than the training set size.")
    train_size = len(full_train_dataset) - args.val_size

    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, args.val_size])
    test_dataset = TrainDataset(csv_paths=args.test_csv_path)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=False, 
        num_workers=args.num_workers
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=args.num_workers
    )


    # training settings
    os.makedirs(args.model_save_dir, exist_ok=True)

    adapter_params = model.dual.parameters()
    adapter_param_ids = {id(p) for p in adapter_params}

    other_trainable_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in adapter_param_ids
    ]

    param_groups = [
        {'params': other_trainable_params, 'lr': args.learning_rate},
        {'params': model.dual.parameters(), 'lr': args.adapter_lr}
    ]

    optimizer = optim.AdamW(
        param_groups,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    num_updates_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_training_steps = num_updates_per_epoch * args.epochs
    num_warmup_steps = num_updates_per_epoch * args.warmup_epochs 


    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - num_warmup_steps,
        eta_min=args.min_lr
    )
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-5,
        total_iters=num_warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[num_warmup_steps]
    )


    # resume the checkpoint
    start_epoch = 0
    global_step = 0

    if args.latest_checkpoint_path:
        print(f"Resuming training from checkpoint: {args.latest_checkpoint_path}")
        checkpoint = torch.load(args.latest_checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")


    # global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):

            text_data = batch['style_prompt']
            audio_input = batch['audio_input']

            loss = model(
                text=text_data, 
                audio_input=audio_input
            )

            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            writer.add_scalar('train/loss', loss.item() * args.gradient_accumulation_steps, global_step)
            writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)


            global_step += 1

            # step testing
            if global_step > 0 and global_step % args.eval_steps == 0 and test_dataloader and val_dataloader:

                evaluate_modify(model, val_dataloader, writer, global_step, 'val')
                evaluate_modify(model, test_dataloader, writer, global_step, 'test')

                # save model
                if epoch + 1 >= args.warmup_epochs:
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    checkpoint_path = os.path.join(args.model_save_dir, f'model_step_{global_step}.pt')
                    torch.save(checkpoint, checkpoint_path)
                

        # epoch testing
        evaluate_modify(model, val_dataloader, writer, global_step, 'val')
        evaluate_modify(model, test_dataloader, writer, global_step, 'test')


    print("Training finished.")
    writer.close()



if '__main__' == __name__:
    main()