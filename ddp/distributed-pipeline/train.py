import os
from torch.distributed.elastic.multiprocessing.errors import record
from config.train import TrainSettings


def parse_args():
    parser = TrainSettings.to_argparse(add_json=True)
    return parser.parse_args()


def train(rank, args):

    # Import dependencies
    import time
    import json

    # Import everything
    from data import load_data_from_args
    from basic_utils import dist_util, logger
    from utils.initialization import create_model_from_config, seed_all
    from utils.trainer import TrainLoop

    dist_util.barrier()  # Sync

    # Set checkpoint path
    folder_name = "model_checkpoints/"
    if not os.path.isdir(folder_name) and rank == 0:
        os.mkdir(folder_name)
    if not args.checkpoint_path:
        model_file = f"Run_{args.dataset}_lr{args.lr}" \
                     f"_seed{args.seed}_{time.strftime('%Y%m%d-%H:%M:%S')}"  # TODO: add your naming rule by args
        args.checkpoint_path = os.path.join(folder_name, model_file)
    if not os.path.isdir(args.checkpoint_path) and rank == 0:
        os.mkdir(args.checkpoint_path)

    # Configure log and seed
    logger.configure(dir=args.checkpoint_path, format_strs=["log", "csv"] + (["stdout"] if rank == 0 else []))
    seed_all(args.seed)

    # Prepare dataloader
    logger.log("### Creating data loader...")
    dist_util.barrier()  # Sync
    data = load_data_from_args(
        split='train',
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        # TODO: add args on your own
        deterministic=False,
        loop=True,
        seed=args.seed,
        num_loader_proc=args.data_loader_workers,
    )
    data_valid = load_data_from_args(
        split='valid',
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        # TODO: add args on your own
        deterministic=True,
        loop=True,
        seed=args.seed,
        num_loader_proc=args.data_loader_workers,
    )
    dist_util.barrier()  # Sync

    # Initialize model
    logger.log("### Creating model...")
    model = create_model_from_config(**args.dict())

    # Load model to each node's device
    model.to(dist_util.dev())
    dist_util.barrier()  # Sync

    # Count and log total params
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'### The parameter count is {pytorch_total_params}')

    # Save training args
    training_args_path = f'{args.checkpoint_path}/training_args.json'
    if not os.path.exists(training_args_path):
        logger.log(f'### Saving the hyperparameters to {args.checkpoint_path}/training_args.json')
        if dist_util.get_rank() == 0:
            with open(training_args_path, 'w') as fp:
                json.dump(args.dict(), fp, indent=2)

    # Init wandb
    if dist_util.get_rank() == 0:
        # TODO: Uncomment and customize your wandb setting on your own, or just use environ.
        import wandb
        wandb.init(
            mode=os.getenv("WANDB_MODE", "online"),
            # entity=os.getenv("WANDB_ENTITY", "<your-value>"),
            # project=os.getenv("WANDB_PROJECT", "<your-value>"),
        )
        wandb.config.update(args.__dict__, allow_val_change=True)
    dist_util.barrier()  # Sync last

    # Run train loop
    logger.log("### Training...")
    train_loop = TrainLoop(
        model=model,
        # TODO: add your argument
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        weight_decay=args.weight_decay,
        learning_steps=args.learning_steps,
        checkpoint_path=args.checkpoint_path,
        gradient_clipping=args.gradient_clipping,
        eval_data=data_valid,
        eval_interval=args.eval_interval,
        eval_callbacks=[]
    )
    train_loop.run_loop()


@record
def main(namespace):

    # Create config from parsed argument namespace
    args: TrainSettings = TrainSettings.from_argparse(namespace)

    # Import dist_util
    from basic_utils import dist_util

    # Setup distributed
    if "LOCAL_RANK" in os.environ:
        dist_util.setup_dist()
    rank = dist_util.get_rank()

    # Run training
    try:
        train(rank, args)
    finally:
        dist_util.cleanup_dist()


if __name__ == "__main__":
    main(parse_args())
