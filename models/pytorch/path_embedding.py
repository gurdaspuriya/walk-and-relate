
"""
KG Path embedding
"""
import copy
import os
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from _thread import start_new_thread
from collections import defaultdict
import traceback
from functools import wraps

from .. import *

norm = lambda x, p: x.norm(p=p)**p

logsigmoid = functional.logsigmoid

def abs(val):
    return th.abs(val)

def masked_select(input, mask):
    return th.masked_select(input, mask)

def get_dev(gpu):
    return th.device('cpu') if gpu < 0 else th.device('cuda:' + str(gpu))

def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))

none = lambda x : x
norm = lambda x, p: x.norm(p=p)**p
get_scalar = lambda x: x.detach().item()
reshape = lambda arr, x, y: arr.view(x, y)
cuda = lambda arr, gpu: arr.cuda(gpu)

def l2_dist(x, y, pw=False):
    if pw is False:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

    return -th.norm(x-y, p=2, dim=-1)

def l1_dist(x, y, pw=False):
    if pw is False:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

    return -th.norm(x-y, p=1, dim=-1)

def dot_dist(x, y, pw=False):
    if pw is False:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

    return th.sum(x * y, dim=-1)

def cosine_dist(x, y, pw=False):
    score = dot_dist(x, y, pw)

    x = x.norm(p=2, dim=-1)
    y = y.norm(p=2, dim=-1)
    if pw is False:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

    return score / (x * y)

def extended_jaccard_dist(x, y, pw=False):
    score = dot_dist(x, y, pw)

    x = x.norm(p=2, dim=-1)**2
    y = y.norm(p=2, dim=-1)**2
    if pw is False:
        x = x.unsqueeze(1)
        y = y.unsqueeze(0)

    return score / (x + y - score)

def floor_divide(input, other):
    return th.floor_divide(input, other)

def thread_wrapped_func(func):
    """Wrapped func for torch.multiprocessing.Process.

    With this wrapper we can use OMP threads in subprocesses
    otherwise, OMP_NUM_THREADS=1 is mandatory.

    How to use:
    @thread_wrapped_func
    def func_to_wrap(args ...):
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

@thread_wrapped_func
def async_update(args, emb, queue):
    """Asynchronous embedding update for entity embeddings.
    How it works:
        1. trainer process push entity embedding update requests into the queue.
        2. async_update process pull requests from the queue, calculate
           the gradient state and gradient and write it into entity embeddings.

    Parameters
    ----------
    args :
        Global confis.
    emb : ExternalEmbedding
        The entity embeddings.
    queue:
        The request queue.
    """
    th.set_num_threads(args.num_thread)
    while True:
        (grad_indices, grad_values, gpu_id) = queue.get()
        clr = emb.args.lr
        if grad_indices is None:
            return
        with th.no_grad():
            grad_sum = (grad_values * grad_values).mean(1)
            device = emb.state_sum.device
            if device != grad_indices.device:
                grad_indices = grad_indices.to(device)
            if device != grad_sum.device:
                grad_sum = grad_sum.to(device)

            emb.state_sum.index_add_(0, grad_indices, grad_sum)
            std = emb.state_sum[grad_indices]  # _sparse_mask
            if gpu_id >= 0:
                std = std.cuda(gpu_id)
            std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
            tmp = (-clr * grad_values / std_values)
            if tmp.device != device:
                tmp = tmp.to(device)
            emb.emb.index_add_(0, grad_indices, tmp)
            
class PathEmbedding:
    """Path Embedding for Knowledge Graph
    Uses multihead attention to project a sequence of relations to an embedding.

    Parameters
    ----------
    args :
        Global configs.
    dim : int
        Embedding dimention size.
    device : th.device
        Device to store the embedding.
    """
    def __init__(self, args, dim, device):
        self.gpu = args.gpu
        self.args = args
        self.dim = dim
        self.num_layers = 1
#         self.sequence_model = nn.RNN(dim, 256, num_layers=self.num_layers, batch_first=True, bidirectional=True, bias=False, nonlinearity='tanh')
#         self.projection = nn.Sequential(nn.Linear(512, 2*dim), nn.Tanh())
#         self.projection2 = nn.Linear(2*dim, dim)
        self.sequence_model = nn.RNN(dim, 256, num_layers=self.num_layers, batch_first=True, bidirectional=True, bias=False, nonlinearity='tanh', device=device)
        self.projection_1 = nn.Linear(512, 2*dim, device=device)
        self.projection_2 = nn.Linear(2*dim, dim, device=device)
        self.dropout = nn.Dropout(p=0.5)
        self.steps = 0.0
        self.device = device
        
        
        
        # optimizer structs for sequence model and projection
        self.state_step = 0
        self.param_group = list(list(self.sequence_model.parameters())+list(self.projection_1.parameters())+list(self.projection_2.parameters()))
        self.state = [{} for i in range(len(self.param_group))]
                
        for i,p in enumerate(self.param_group):
            state = self.state[i]
            state['step'] = 0
            init_value = 0
            state['sum'] = th.full_like(p, init_value, memory_format=th.preserve_format)
            
#         self.optimizer = th.optim.Adam(self.param_group, lr=args.rnn_lr, weight_decay=args.regularization_coef, betas=(0.9, 0.5))
#         self.optimizer = th.optim.Adam(self.param_group, lr=args.rnn_lr, weight_decay=args.regularization_coef, betas=(0.9, 0.999))
        self.optimizer = th.optim.Adam(self.param_group, lr=args.rnn_lr, weight_decay=args.regularization_coef)
            
    def init(self, emb_init):
        """Initializing the model.

        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
#         INIT.eye_(self.sequence_model.weight_hh_l0)
#         INIT.eye_(self.sequence_model.weight_ih_l0)
#         INIT.eye_(self.sequence_model.weight_hh_l0_reverse)
#         INIT.eye_(self.sequence_model.weight_ih_l0_reverse)
        gain = INIT.calculate_gain('tanh')
        for i in range(self.num_layers):
            INIT.xavier_normal_(getattr(self.sequence_model, 'weight_hh_l{}'.format(i)), gain=gain)
            INIT.xavier_normal_(getattr(self.sequence_model, 'weight_ih_l{}'.format(i)), gain=gain)
            INIT.xavier_normal_(getattr(self.sequence_model, 'weight_hh_l{}_reverse'.format(i)), gain=gain)
            INIT.xavier_normal_(getattr(self.sequence_model, 'weight_ih_l{}_reverse'.format(i)), gain=gain)
#             INIT.xavier_normal_(self.sequence_model.weight_hh_l0, gain=INIT.calculate_gain('tanh'))
#             INIT.xavier_normal_(self.sequence_model.weight_ih_l0, gain=INIT.calculate_gain('tanh'))
#             INIT.xavier_normal_(self.sequence_model.weight_hh_l0_reverse, gain=INIT.calculate_gain('tanh'))
#             INIT.xavier_normal_(self.sequence_model.weight_ih_l0_reverse, gain=INIT.calculate_gain('tanh'))
        INIT.xavier_normal_(self.projection_1.weight, gain=gain)
#         INIT.uniform_(self.projection.weight[0], -emb_init/(self.dim*100), emb_init/(self.dim*100))
#         self.sequence_model = self.sequence_model.to(self.device)
#         self.projection_1 = self.projection.to(self.device)
#         self.projection2 = self.projection2.to(self.device)
        
        
    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process tensor access
        """
#         self.sequence_model.share_memory()
#         self.projection.share_memory()
        
        for state in self.state:
            state['sum'].share_memory_()

    def __call__(self, relation_embeddings, path_lengths, direction, gpu_id=-1, trace=True):
        """ Return path embedings.

        Parameters
        ----------
        relation_embeddings : th.tensor
            individual embeddings of the relation: batch_size x seq_length x dim
        path_lengths : th.tensor
            Length of the paths: batch_size
        direction : th.tensor
            direction of the relations: batch_size x seq_length
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
#         sequence_model = self.sequence_model
#         projection = self.projection
#         if gpu_id >= 0:
#             path_lengths = path_lengths.cuda(gpu_id)
#             direction = direction.cuda(gpu_id)
#             sequence_model = copy.deepcopy(self.sequence_model).cuda(gpu_id)
#             projection = copy.deepcopy(self.projection).cuda(gpu_id)
#         else:
#             sequence_model = self.sequence_model
#             projection = self.projection
        if trace is False:
            with th.no_grad():
#                 self.projection_1.eval()
                self.sequence_model.eval()
#                 relation_embeddings = th.cat((relation_embeddings, direction.unsqueeze(-1)), dim=-1)
                relation_embeddings = direction.float().unsqueeze(-1).expand_as(relation_embeddings)*relation_embeddings
                path_lengths, perm_idx = th.sort(path_lengths, dim=0, descending=True, stable=True)
                _, reverse_perm_idx = th.sort(perm_idx, stable=True)
                relation_embeddings = th.tanh(relation_embeddings[perm_idx])
#                 relation_embeddings = relation_embeddings[perm_idx]
                packed_input = pack_padded_sequence(relation_embeddings, path_lengths.cpu().numpy(), batch_first=True)
                packed_output, ht = self.sequence_model(packed_input)
                result = self.projection_1(th.cat((ht[0,:,:], ht[1,:,:]), dim=-1)[reverse_perm_idx])
                result = self.projection_2(th.tanh(result))
#                 self.projection.train()
                self.sequence_model.train()
                return result
        else:
            
#             relation_embeddings = th.cat((relation_embeddings, direction.unsqueeze(-1)), dim=-1)
#             self.param_group_device = list(list(sequence_model.parameters())+list(projection.parameters()))
            relation_embeddings = direction.float().unsqueeze(-1).expand_as(relation_embeddings)*relation_embeddings
            path_lengths, perm_idx = th.sort(path_lengths, dim=0, descending=True, stable=True)
            _, reverse_perm_idx = th.sort(perm_idx, stable=True)
            relation_embeddings = th.tanh(relation_embeddings[perm_idx])
#             relation_embeddings = relation_embeddings[perm_idx]
            packed_input = pack_padded_sequence(relation_embeddings, path_lengths.cpu().numpy(), batch_first=True)
            packed_output, ht = self.sequence_model(packed_input)
            result = self.projection_1(th.cat((ht[0,:,:], ht[1,:,:]), dim=-1))[reverse_perm_idx]
            result = self.projection_2(self.dropout(th.tanh(result)))
#             for param in self.param_group:
#                 print (param.data)
#             print(relation_embeddings, packed_output, ht, reverse_perm_idx, result)
            return result
#         if gpu_id >= 0:
#             path_lengths = path_lengths.cuda(gpu_id)
#             direction = direction.cuda(gpu_id)
#             self.sequence_model = self.sequence_model.cuda(gpu_id)
#             self.projection = self.projection.cuda(gpu_id)
#         if trace is False:
#             with th.no_grad():
# #                 relation_embeddings = th.cat((relation_embeddings, direction.unsqueeze(-1)), dim=-1)
#                 relation_embeddings = direction.float().unsqueeze(-1).expand_as(relation_embeddings)*relation_embeddings
#                 path_lengths, perm_idx = path_lengths.sort(0, descending=True)
#                 reverse_perm_idx = th.argsort(perm_idx)
#                 relation_embeddings = relation_embeddings[perm_idx]
#                 packed_input = pack_padded_sequence(relation_embeddings, path_lengths.cpu().numpy(), batch_first=True)
#                 packed_output, ht = self.sequence_model(packed_input)
#                 return self.projection(th.cat((ht[0,:,:], ht[1,:,:]), dim=-1)[reverse_perm_idx])
#         else:
# #             relation_embeddings = th.cat((relation_embeddings, direction.unsqueeze(-1)), dim=-1)
#             relation_embeddings = direction.float().unsqueeze(-1).expand_as(relation_embeddings)*relation_embeddings
#             path_lengths, perm_idx = path_lengths.sort(0, descending=True)
#             reverse_perm_idx = th.argsort(perm_idx)
#             relation_embeddings = relation_embeddings[perm_idx]
#             packed_input = pack_padded_sequence(relation_embeddings, path_lengths.cpu().numpy(), batch_first=True)
#             packed_output, ht = self.sequence_model(packed_input)
#             return self.projection(th.cat((ht[0,:,:], ht[1,:,:]), dim=-1)[reverse_perm_idx])
            

    def update(self, weight_decay = 0.0, norm_p = 3, gpu_id=-1):
        """ Update model.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
#         norm_p = 1
        self.state_step += 1
        clr = self.args.rnn_lr
        self.steps += 1
        self.optimizer.step()
        self.optimizer.zero_grad()
#         device = self.state[0]['sum'].device
#         with th.no_grad():
#             for i,p in enumerate(self.param_group_device):
#                 if p.grad is None:
#                     continue
#                 if weight_decay != 0 and norm_p > 0:
#                     p.grad = p.grad + 0.0*weight_decay*norm_p*th.pow(th.abs(p.data), norm_p-1)*th.sign(p.data)
                    
#                 temp_p_grad = p.grad.to(device)
# # #                 print(p, p.grad)
# #                 if p.device != device:
# #                     p.data = p.to(device)
# #                     p.grad.data = p.grad.to(device)
# #                 print(p, p.grad)
#                 self.state[i]['sum'].addcmul_(temp_p_grad, temp_p_grad, value=1)
#                 std = self.state[i]['sum'].sqrt().add_(1e-15)
#                 self.param_group[i].addcdiv_(temp_p_grad, std, value=-clr)

#                 # Reset the grads
#                 if p.grad.grad_fn is not None:
#                     p.grad.detach_()
#                 else:
#                     p.grad.requires_grad_(False)
#                 p.grad.zero_()


    def create_async_update(self):
        """Set up the async update subprocess.
        """
        self.async_q = Queue(1)
        self.async_p = mp.Process(target=async_update, args=(self.args, self, self.async_q))
        self.async_p.start()

    def finish_async_update(self):
        """Notify the async update subprocess to quit.
        """
        self.async_q.put((None, None, None))
        self.async_p.join()

    def curr_emb(self):
        """Return embeddings in trace.
        """
        data = [data for _, data, mask in self.trace]
        return th.cat(data, 0)

    def save(self, path, name):
        """Save embeddings.

        Parameters
        ----------
        path : str
            Directory to save the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name+'.npy')
        np.save(file_name, self.emb.cpu().detach().numpy())

    def load(self, path, name):
        """Load embeddings.

        Parameters
        ----------
        path : str
            Directory to load the embedding.
        name : str
            Embedding name.
        """
        file_name = os.path.join(path, name+'.npy')
        self.emb = th.Tensor(np.load(file_name))
