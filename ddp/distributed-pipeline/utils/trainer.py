"""
Trainer copied and adapted from:
https://github.com/Shark-NLP/DiffuSeq/blob/8bfafcbb26df218073b8117234afb9de9dfcbec9/train_util.py
"""
import copy
import os

import blobfile as bf
import math
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import AdamW

from basic_utils import dist_util, logger


class TrainLoop:

    def log_loss_dict(self, mode, losses, *args, **kwargs):  # mode: train or eval
        # TODO: Add your loss logging function with logger.logkv_mean
        raise NotImplementedError

    def compute_losses(self, micro_batch):
        # TODO: Add your loss function with logger.logkv_mean
        raise NotImplementedError

    @staticmethod
    def backward_from_losses(losses):
        # TODO: Add your loss-reducing function
        loss = ...
        loss.backward()

    @classmethod
    def get_batch_length(cls, batch):
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        elif isinstance(batch, dict):
            return cls.get_batch_length(batch[list(batch.keys())[0]])
        elif isinstance(batch, list):
            return cls.get_batch_length(batch[0])
        else:
            # TODO: Add your custom function if needed
            raise TypeError("Unsupported batch type: {}".format(type(batch).__name__))

    def __init__(
            self,
            *,
            model,
            # TODO: Add your arguments
            data,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            weight_decay=0.0,
            learning_steps=0,
            checkpoint_path='',
            gradient_clipping=-1.,
            eval_data=None,
            eval_interval=-1,
            eval_callbacks=(),  # e.g. plotting utils for wandb
    ):
        self.model = model
        self.data = data
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = float(lr)
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.weight_decay = weight_decay
        self.learning_steps = learning_steps
        self.gradient_clipping = gradient_clipping

        # TODO: Add your arguments

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist_util.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.eval_callbacks = list(eval_callbacks)

        self.checkpoint_path = checkpoint_path  # DEBUG **

        self._load_and_sync_parameters()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self._load_optimizer_state()
            # frac_done = (self.step + self.resume_step) / self.learning_steps
            # lr = self.lr * (1 - frac_done)
            # self.opt = AdamW(self.master_params, lr=lr, weight_decay=self.weight_decay)
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if dist_util.is_initialized():
            self.use_ddp = True
            print(dist_util.dev())
            ddp_kwargs = dict(
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            if not torch.cuda.is_available():
                ddp_kwargs.update(device_ids=[dist_util.dev()])
                ddp_kwargs.update(output_device=dist_util.dev())
            self.ddp_model = DistributedDataParallel(self.model, **ddp_kwargs)
        else:
            self.use_ddp = False
            self.ddp_model = self.model

        # In checkpoint-loading process, sometimes GPU 0 is used and allocated.
        # After all process is done, free GPU 0 by function below:
        torch.cuda.empty_cache()

    # # # # # # # # # # # # # # # Implemented Functions # # # # # # # # # # # # # # #

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        if not resume_checkpoint:
            return

        self.resume_step = self.parse_resume_step_from_filename(resume_checkpoint)
        if dist_util.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
            )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        if not main_checkpoint:
            return
        ema_checkpoint = self.find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist_util.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=dist_util.dev())
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint
        if not main_checkpoint:
            return
        opt_checkpoint = self.find_opt_checkpoint(main_checkpoint, self.resume_step)
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.learning_steps
            or self.step + self.resume_step < self.learning_steps
        ):
            batch = next(self.data)
            self.run_step(batch)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.eval_data is not None and self.step % self.eval_interval == 0:
                cond_eval = next(self.eval_data)
                self.forward_only(cond_eval)
                print('eval on validation set')
                logger.dumpkvs()
                if dist_util.get_rank() == 0:
                    for callback in self.eval_callbacks:
                        callback(self)
            if self.step > 0 and self.step % self.save_interval == 0:
                self.save()
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        self.optimize()
        self.log_step()

    def _zero_grad(self):
        for param in self.model_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def _common_forward(self, total_batch, start_index):
        micro_batch = {
            k: v[start_index: start_index + self.microbatch].to(dist_util.dev())
            for k, v in total_batch.items()
        }
        last_batch = (start_index + self.microbatch) >= self.get_batch_length(total_batch)

        if last_batch or not self.use_ddp:
            losses = self.compute_losses(micro_batch)
        else:
            with self.ddp_model.no_sync():
                losses = self.compute_losses(micro_batch)
        return losses

    @torch.no_grad()
    def forward_only(self, batch):
        self._zero_grad()
        for i in range(0, self.get_batch_length(batch), self.microbatch):
            losses = self._common_forward(batch, i)
            self.log_loss_dict(mode="eval", losses=losses)

    def forward_backward(self, batch):
        self._zero_grad()
        for i in range(0, self.get_batch_length(batch), self.microbatch):
            losses = self._common_forward(batch, i)
            self.log_loss_dict(mode="eval", losses=losses)
            self.backward_from_losses(losses)

    def optimize(self):
        if self.gradient_clipping > 0:
            self.grad_clip()
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def grad_clip(self):
        max_grad_norm = self.gradient_clipping
        if hasattr(self.opt, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            self.opt.clip_grad_norm(max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm,
            )

    def _anneal_lr(self):
        if not self.learning_steps:
            return
        frac_done = (self.step + self.resume_step) / self.learning_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def _log_grad_norm(self):
        sqsum = 0.0
        # cnt = 0
        for p in self.master_params:
            if p.grad is not None:
                sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", math.sqrt(sqsum))

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        self._save_checkpoint(0, self.master_params)
        for r, p in zip(self.ema_rate, self.ema_params):
            self._save_checkpoint(r, p)
        self._save_opt()
        dist_util.barrier()

    def _save_checkpoint(self, rate, params):
        state_dict = self._master_params_to_state_dict(params)
        if dist_util.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model_{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            print('writing to', bf.join(self.checkpoint_path, filename))
            with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                torch.save(state_dict, f)

    def _save_opt(self):
        if dist_util.get_rank() == 0:
            logger.log(f"saving optimizer...")
            filename = f"opt_{(self.step+self.resume_step):06d}.pt"
            print('writing to', bf.join(self.checkpoint_path, filename))
            with bf.BlobFile(bf.join(self.checkpoint_path, filename), "wb") as f:
                torch.save(self.opt.state_dict(), f)

    def _master_params_to_state_dict(self, master_params, key=None):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            if key is not None and key == name:
                return master_params[i]
            state_dict[name] = master_params[i]
        if key is not None:
            raise KeyError(key)
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return params

    @staticmethod
    def parse_resume_step_from_filename(filename):
        """
        Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        filename: str = os.path.basename(filename)
        assert filename.startswith('model') and filename[-3:] == '.pt', "Invalid model name"
        return int(filename[-9:-3])

    @staticmethod
    def find_resume_checkpoint():
        log_dir = logger.get_current().dir
        model_weights = sorted(filter(lambda s: s.endswith(".pt") and s.startswith("model"), os.listdir(log_dir)))
        if model_weights:
            path = os.path.join(log_dir, model_weights[-1])
            return path

    @staticmethod
    def find_ema_checkpoint(main_checkpoint, step, rate):
        if not main_checkpoint:
            return None
        filename = f"ema_{rate}_{step:06d}.pt"
        path = bf.join(bf.dirname(main_checkpoint), filename)
        if bf.exists(path):
            return path
        return None

    @staticmethod
    def find_opt_checkpoint(main_checkpoint, step):
        if not main_checkpoint:
            return None
        filename = f"opt_{step:06d}.pt"
        path = bf.join(bf.dirname(main_checkpoint), filename)
        if bf.exists(path):
            return path
        return None

    __call__ = run_loop


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for trg, src in zip(target_params, source_params):
        trg.detach().mul_(rate).add_(src, alpha=1 - rate)
