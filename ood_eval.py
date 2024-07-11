import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dataclasses import dataclass
from utils import tokenize_and_concatenate
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import *
from pandas import DataFrame


@dataclass
class OODLensEvaluatorConfig:
    dataset = 'Skylion007/openwebtext'

    seed = 0
    batch_size = 8
    context_len = 128

    eval_type = 'add_random_acts'

    is_transcoder = True
    sae_layer = 8
    hook_in_name = 'blocks.8.ln2.hook_normalized'
    hook_out_name = 'blocks.8.hook_mlp_out'
    sae_out_fn =  None

    device = 'mps'

class OODLensEvaluator():
    def __init__(self, cfg, model, sae, feature_mask):
        self.cfg = cfg
        self.model = model #TransformerLens Hooked Model
        self.sae = sae
        self.feature_mask = feature_mask #features to add
        self.device = cfg.device

    @torch.no_grad()
    def evaluate(self, eval_type = None, tokens = 5e6):
        if eval_type is None:
            eval_type = self.cfg.eval_type

        dataloader = self.get_dataloader()
        steps = int(tokens/(self.cfg.context_len * self.cfg.batch_size))+1
        pbar = tqdm(range(steps))

        self.set_input_decoder()

        self.metrics = []
        for _ in pbar:
            batch = next(dataloader)
            batch['labels'] = batch['input_ids']

            #x: sae inputs, y: sae outputs, loss: cross-entropy loss
            x, y, y_hat, sae_acts, logits = self.run_model(batch) 
            logits_hat, _ = self.patch_model(batch, outputs=y_hat)
            mse = ((y - y_hat)**2).sum(dim=-1).mean()
            y_sq = (y**2).sum(dim=-1).mean()

            self.metrics.append({'k': 0,
                       'mse': mse.item(),
                       'mse_ratio': mse.item() / y_sq.item(),
                       'kl_div': self.kl_div(logits, logits_hat),
                       'L0': (sae_acts>0).float().sum(dim=-1).mean().item()
                       })

            for k in [1, 3, 5, 10, 20, 50]:
                x_ood, y_ood_hat, sae_acts_ood = self.get_ood_inputs(x, sae_acts, eval_type, k)
                logits_ood, y_ood = self.patch_model(batch, inputs=x_ood)
                logits_ood_hat, _ = self.patch_model(batch, inputs=x_ood, outputs=y_ood_hat)

                mse_ood = ((y_ood - y_ood_hat)**2).sum(dim=-1).mean()
                y_sq_ood = (y_ood**2).sum(dim=-1).mean()
                self.metrics.append({'k': k,
                       'mse': mse_ood.item(),
                       'mse_ratio': mse_ood.item() / y_sq_ood.item(),
                       'kl_div': self.kl_div(logits_ood, logits_ood_hat),
                       'L0': (sae_acts_ood>0).float().sum(dim=-1).mean().item(),
                       'Diff L0': ((sae_acts_ood > 0) & (sae_acts==0)).float().sum(dim=-1).mean().item()
                       })
        self.metrics = DataFrame(self.metrics)
        return self.metrics.groupby('k').mean()


    @torch.no_grad()
    def run_model(self, batch):
        logits, cache = self.model.run_with_cache(batch['input_ids'], 
                                            names_filter=[self.cfg.hook_in_name,
                                                            self.cfg.hook_out_name])
        inputs = cache[self.cfg.hook_in_name]
        outputs = cache[self.cfg.hook_out_name]
        outputs_hat, sae_acts = self.cfg.sae_out_fn(inputs)
        del cache
        return inputs, outputs, outputs_hat, sae_acts, logits

    @torch.no_grad()
    def get_ood_inputs(self, x, sae_acts, eval_type, k):
        if eval_type == 'add_random_acts':
            size = list(sae_acts.shape)
            size[-1] = k
            rand_ids = self.get_rand_ids(size, feature_mask=self.feature_mask)
            sample_acts = self.sample_activations(sae_acts, rand_ids)
            x_add = einsum(self.W_dec_in[rand_ids], sample_acts, "b pos feat d_model, b pos feat -> b pos d_model")
            x_ood = self.add_to_inputs(x, x_add)
        elif eval_type == 'add_noise':
            sigma = k * 0.001 * x.norm(dim=-1).mean()
            x_add = sigma * torch.randn(x.shape).to(self.cfg.device)
            x_ood = self.add_to_inputs(x, x_add)

        y_ood_hat, sae_acts_ood = self.cfg.sae_out_fn(x_ood)
        return x_ood, y_ood_hat, sae_acts_ood

    @torch.no_grad()
    def patch_model(self, batch, inputs = None, outputs=None):
        hooks = []
        if inputs is not None:
            hooks.append((self.cfg.hook_in_name, 
                lambda acts, hook: inputs))

        if outputs is not None:
            hooks.append((self.cfg.hook_out_name, 
                lambda acts, hook: outputs))

        with self.model.hooks(fwd_hooks=hooks):
            logits, cache = self.model.run_with_cache(batch['input_ids'], 
                                names_filter=[self.cfg.hook_out_name])
            y = cache[self.cfg.hook_out_name]
        return logits, y

    def get_rand_ids(self, size, feature_mask = None):
        feat_ids = torch.arange(self.sae.W_dec.shape[0])
        if feature_mask is not None:
            feat_ids = feat_ids[feature_mask]
        rand_choice = torch.randint(len(feat_ids), size = size)
        rand_ids = feat_ids[rand_choice].to(self.device)
        return rand_ids

    # @torch.no_grad()
    def sample_activations(self, sae_acts, rand_ids):
        acts = sae_acts[sae_acts > 0]
        ids = torch.randint(len(acts), size=rand_ids.shape)
        sample_acts = acts[ids].to(self.cfg.device) 
        # sample_acts += - self.sae.b_enc[rand_ids] #so that SAE activations = sample_acts
        return sample_acts

    def add_to_inputs(self, x, x_add):
        x_ood = x + x_add
        x_ood = x_ood * x.norm(dim=-1, keepdim=True) / x_ood.norm(dim=-1, keepdim=True)
        return x_ood

    @torch.no_grad()
    def set_input_decoder(self):
        if self.cfg.is_transcoder:
            # find W_dec_in so that W_dec_in @ W_enc is approx Identity
            W_enc = self.sae.W_enc.detach()
            self.W_dec_in = W_enc.T / (W_enc.T.norm(dim=-1,keepdim=True)**2)
            
            # b_dec = self.sae.b_dec.detach()
            # proj = torch.eye(W_enc.shape[0],W_enc.shape[0]).to(self.cfg.device) - b_dec @ b_dec.T
            # W_dec_in = (proj @ W_enc).T
            # denom = torch.diag(W_dec_in @ W_enc)
            # self.W_dec_in = torch.diag(1/denom) @ W_dec_in
        else:
            self.W_dec_in = self.sae.W_dec.detach()

    @torch.no_grad()
    def kl_div(self, logits, logits_hat):
        log_probs = F.log_softmax(logits_hat, dim=-1)
        target = F.softmax(logits, dim=-1)
        kl_div = F.kl_div(log_probs, target, reduction='batchmean')
        return kl_div.item()

    def get_dataset(self):
        dataset = load_dataset(self.cfg.dataset, split='train', streaming=True,
                                trust_remote_code=True)
        dataset = dataset.shuffle(seed=self.cfg.seed, buffer_size=10_000)
        tokenized_dataset = tokenize_and_concatenate(dataset, self.model.tokenizer, 
                            max_length=self.cfg.context_len, streaming=True
                            )
        return tokenized_dataset

    def get_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.cfg.batch_size
        dataset = self.get_dataset()
        return iter(DataLoader(dataset, batch_size=batch_size))



class OODEvaluatorNNsight(OODLensEvaluator):

    @torch.no_grad()
    def run_model(self):
        with self.model.trace(batch):
            input = module_fn(model).input[0][0].save()
            if self.cfg.is_transcoder:
                output = hook_out_fn(model).output.save()
            ce_loss = model.output.loss.save()
        y = output.value if self.cfg.is_transcoder else input.value
        y_hat, sae_acts = self.cfg.sae_out_fn(input.value)
        return y, y_hat, sae_acts, ce_loss.value

    @torch.no_grad()
    def patch_model(self, batch, y):
        with self.model.trace(batch):
            if self.cfg.is_transcoder:
                setattr(module_fn(model), 'output', y)
            else:
                setattr(module_fn(model), 'input', y)
            ce_loss = model.output.loss.save()
        return ce_loss.value