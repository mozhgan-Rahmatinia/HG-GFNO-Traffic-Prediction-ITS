import os, time, shutil, argparse, configparser, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.models import make_model
from lib.utils import (get_adjacency_matrix, compute_val_loss_GFNO,
                     predict_and_save_results_GFNO, EarlyStopping,
                     adjust_learning_rate)
from tensorboardX import SummaryWriter
from lib.metrics import masked_mae, masked_mse
from data_provider.data_factory import data_provider
import sys


class Tee:

    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode, encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# ---------- Utilities ---------- #
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def _get_data(root, flag, seq, lab, pred, bs):
    return data_provider(root, flag, seq, lab, pred, bs)
def get_lr(opt): return opt.param_groups[0]['lr']


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_prediction_plot(writer, epoch, model, data_loader, device, n_nodes):
    
    model.eval()
    with torch.no_grad():
        
        x, y_true = next(iter(data_loader))
        x, y_true = x.float().to(device), y_true.float().to(device)
        y_pred, _ = model(x)

       
        sample_idx = random.randint(0, x.shape[0] - 1)
        node_idx = random.randint(0, n_nodes - 1)
        
        true_vals = y_true[sample_idx, :, node_idx].cpu().numpy()
        pred_vals = y_pred[sample_idx, :, node_idx].cpu().numpy()

        fig = plt.figure(figsize=(12, 6))
        plt.plot(true_vals, label='Ground Truth', color='blue', marker='o', linestyle='-')
        plt.plot(pred_vals, label='Prediction', color='red', marker='x', linestyle='--')
        plt.title(f'Prediction vs. Ground Truth (Sample {sample_idx}, Node {node_idx}) at Epoch {epoch}')
        plt.xlabel('Time Step (Prediction Horizon)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        writer.add_figure(f'Predictions/Comparison', fig, global_step=epoch)
        plt.close(fig)
# -------------------------------- #

def train_main(args):
    # -------- read config (UTF-8 + inline comments) --------
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read(args.config, encoding='utf-8')
    data_cfg, train_cfg = cfg['Data'], cfg['Training']

    # ---------- data info ----------
    adj_file  = data_cfg['adj_filename']
    gnpz      = data_cfg['graph_signal_matrix_filename']
    n_nodes   = int(data_cfg['num_of_vertices'])
    pph       = int(data_cfg['points_per_hour'])
    pred_len  = int(data_cfg['num_for_predict'])
    len_input = int(data_cfg['len_input'])
    dataset   = data_cfg['dataset_name']
    id_file   = data_cfg.get('id_filename', None)

    # ---------- training hyper-params ----------
    ctx       = train_cfg['ctx']; os.environ['CUDA_VISIBLE_DEVICES'] = ctx
    DEVICE    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lr0       = float(train_cfg['learning_rate'])
    epochs    = int(train_cfg['epochs'])
    bs        = int(train_cfg['batch_size'])
    in_ch     = int(train_cfg['in_channels'])
    nb_chev   = int(train_cfg['nb_chev_filter'])
    nb_time   = int(train_cfg['nb_time_filter'])
    K         = int(train_cfg['K'])
    loss_nm   = train_cfg['loss_function']
    metric_m  = train_cfg['metric_method']
    miss_val  = float(train_cfg['missing_value'])
    time_str  = int(train_cfg.get('time_strides', 1))
    wdecay    = float(train_cfg.get('weight_decay', 0.0))
    grad_clip = float(train_cfg.get('grad_clip', 0.0))
    sched_tp  = train_cfg.get('scheduler', 'plateau')      # step | plateau
    lr_step   = int(train_cfg.get('lr_step', 10))
    lr_decay  = float(train_cfg.get('lr_decay', 0.5))
    patience  = int(train_cfg.get('patience', 10))
    seed      = int(train_cfg.get('seed', 2024))
    print_it  = int(train_cfg.get('print_iter_every', 50))

    set_seed(seed)
    print(f"CUDA={torch.cuda.is_available()}  device={DEVICE}")

    # ---------- dirs ----------
    save_dir = os.path.join('experiments', dataset, f'predict{pred_len}_MGCN_{time.strftime("%Y%m%d_%H%M%S")}')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    log_filepath = os.path.join(save_dir, 'training_log.txt')
    logger = Tee(log_filepath)

    
    writer = SummaryWriter(logdir=save_dir, flush_secs=5)

    # ---------- data ----------
    tr_set, tr_loader = _get_data(gnpz, 'train', len_input, 0, pred_len, bs)
    va_set, va_loader = _get_data(gnpz, 'val',   len_input, 0, pred_len, bs)
    te_set, te_loader = _get_data(gnpz, 'test',  len_input, 0, pred_len, bs)
    print(f"Data Loaded: Train {len(tr_set)}, Val {len(va_set)}, Test {len(te_set)}")

    # ---------- graph & model ----------
    adj_mx, _ = get_adjacency_matrix(adj_file, n_nodes, id_file)
    net = make_model(DEVICE, in_ch, K, nb_chev, nb_time, time_str,
                     adj_mx, pred_len, len_input).to(DEVICE)
    print(net)
    

    # # add
    # # *** Get one sample batch to trace the graph ***
    # sample_batch, _ = next(iter(tr_loader))
    # sample_batch = sample_batch.float().to(DEVICE)

    # # *** Add the model graph to TensorBoard ***
    # writer.add_graph(net, sample_batch)
    # # end add

    num_params = count_parameters(net)
    print(f"Total Trainable Parameters: {num_params:,}")

    hparams = {
        "learning_rate": lr0, "epochs": epochs, "batch_size": bs,
        "in_channels": in_ch, "nb_chev_filter": nb_chev, "nb_time_filter": nb_time,
        "K": K, "loss_function": loss_nm, "weight_decay": wdecay,
        "grad_clip": grad_clip, "scheduler": sched_tp, "lr_step": lr_step,
        "lr_decay": lr_decay, "patience": patience, "seed": seed,
        "dataset": dataset, "len_input": len_input, "num_for_predict": pred_len,
        "ctx": ctx, "trainable_params": num_params
    }

    writer.add_hparams(hparams, {})
    
    print("\n----------- Hyperparameters -----------")
    for key, value in hparams.items():
        print(f"- {key}: {value}")
    print("---------------------------------------\n")


    # ---------- loss ----------
    masked = loss_nm.startswith('masked')
    crit_m = masked_mae if loss_nm == 'masked_mae' else masked_mse
    crit   = nn.L1Loss() if 'mae' in loss_nm else nn.MSELoss()
    crit = crit.to(DEVICE)

    # ---------- optim / sched ----------
    optimiser = optim.Adam(net.parameters(), lr=lr0, weight_decay=wdecay)
    plateau = None
    if sched_tp == 'plateau':
        plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode='min', factor=lr_decay,
            patience=max(1, lr_step//2), verbose=True, min_lr=1e-5
        )

    stopper = EarlyStopping(patience=patience)

    best_val = 1e9; best_ep = 0
    global_step = 0
    for ep in range(epochs):
        net.train(); ep_loss = 0.0; epoch_start_time = time.time()
        for bi, (x, y) in enumerate(tr_loader, 1):
            iter_start_time = time.time()
            global_step += 1
            
            x, y = x.float().to(DEVICE), y.float().to(DEVICE)
            optimiser.zero_grad()
            out, _ = net(x)
            loss = crit_m(out, y, miss_val) if masked else crit(out, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimiser.step()
            ep_loss += loss.item()

            # *** لاگ کردن اطلاعات در TensorBoard ***
            iter_time = time.time() - iter_start_time
            writer.add_scalar('Loss/train_iteration', loss.item(), global_step)
            writer.add_scalar('Performance/iteration_time_sec', iter_time, global_step)
            if torch.cuda.is_available():
                writer.add_scalar('Performance/gpu_mem_allocated_MB', 
                                  torch.cuda.memory_allocated() / (1024 * 1024 * 1024), global_step)
                writer.add_scalar('Performance/gpu_mem_cachedd_MB', 
                                  torch.cuda.memory_cached() / (1024 * 1024 * 1024), global_step)
                writer.add_scalar('Performance/gpu_mem_total_MB', 
                                  torch.cuda.memory_allocated() + torch.cuda.memory_allocated() / (1024 * 1024 * 1024), global_step)
            if bi % print_it == 0:
                print(f"Ep {ep+1} | it {bi} | loss {loss.item():.4f} | lr {get_lr(optimiser):.3e}")
        
        ep_loss /= bi
        epoch_time = time.time() - epoch_start_time
        
        # *** لاگ کردن اطلاعات پایان اپک ***
        writer.add_scalar('Loss/train_epoch', ep_loss, ep + 1)
        writer.add_scalar('Performance/epoch_time_sec', epoch_time, ep + 1)

        # validation
        val_loss = compute_val_loss_GFNO(
            net, va_loader, crit_m if masked else crit,
            masked, miss_val, writer, ep, DEVICE
        )
        
        writer.add_scalar('Loss/validation_epoch', val_loss, ep + 1)
        
        # # add
        # if node_embeddings is not None:
        #     writer.add_embedding(
        #         mat=node_embeddings,
        #         # A unique tag helps compare embeddings across epochs
        #         tag=f'Node_Embeddings_Epoch_{ep+1}',
        #         global_step=ep + 1
        #     )
        # # end add

        # add
        for name, param in net.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Calculate the L2 norm of the gradient for each parameter tensor
                grad_norm = param.grad.norm(2)
                writer.add_scalar(f'Gradients_Norm/{name}', grad_norm, ep + 1)
        # end add

        print(f"Ep {ep+1}/{epochs}  train {ep_loss:.4f} | val {val_loss:.4f} | epoch time {epoch_time:.2f}s")

        # scheduler
        if sched_tp == 'plateau':
            plateau.step(val_loss)
        else:  # step
            adjust_learning_rate(optimiser, ep+1, lr0, decay=lr_decay, step=lr_step)

        # early-stop
        ckpt = os.path.join(save_dir, f'epoch_{ep+1}.params')
        stopper(val_loss, net, ckpt)
        if val_loss < best_val:
            best_val, best_ep = val_loss, ep + 1
            shutil.copyfile(ckpt, os.path.join(save_dir, 'ckpt_best.params'))
            
            log_prediction_plot(writer, ep + 1, net, va_loader, DEVICE, n_nodes)

        if stopper.early_stop:
            print("Early stopping"); break

    print(f"Best epoch {best_ep}  val {best_val:.4f}")
    predict_main(best_ep, te_loader, te_set, pred_len, metric_m, save_dir, net)
    writer.close()

# ---------- test ----------
def predict_main(step, loader, data, pred_len, metric, path, net):
    f = os.path.join(path, f'epoch_{step}.params')
    net.load_state_dict(torch.load(f, map_location=next(net.parameters()).device))
    predict_and_save_results_GFNO(net, loader, data, pred_len,
                                  step, metric, path, 'test')

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configurations/PEMS08_mgcn.conf")
    return p.parse_args()

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args(); train_main(args)
