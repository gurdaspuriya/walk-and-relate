# -*- coding: utf-8 -*-
#
# tensor_models.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
KG Sparse embedding
"""
import os
import numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT

import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from _thread import start_new_thread
import traceback
from functools import wraps

from .. import *

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

class InferEmbedding:
    def __init__(self, device):
        self.device = device

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

    def load_emb(self, emb_array):
        """Load embeddings from numpy array.

        Parameters
        ----------
        emb_array : numpy.array  or torch.tensor
            Embedding array in numpy array or torch.tensor
        """
        if isinstance(emb_array, np.ndarray):
            self.emb = th.Tensor(emb_array)
        else:
            self.emb = emb_array

    def __call__(self, idx):
        return self.emb[idx].to(self.device)

class ExternalEmbedding:
    """Sparse Embedding for Knowledge Graph
    It is used to store both entity embeddings and relation embeddings.

    Parameters
    ----------
    args :
        Global configs.
    num : int
        Number of embeddings.
    dim : int
        Embedding dimention size.
    device : th.device
        Device to store the embedding.
    """
    def __init__(self, args, num, dim, device, num_rels=-1):
        self.gpu = args.gpu
        self.args = args
        self.num = num
        self.dim = dim
        self.device = device
        self.trace = []

        if num_rels > 0 and num_rels < num:
            self.basis = ExternalEmbedding(args, 1, num_rels*dim, device, -1)
            self.emb = th.empty(num, num_rels, dtype=th.float32, device=device)
        else:
            self.basis = None
            self.emb = th.empty(num, dim, dtype=th.float32, device=device)
            
#         self.emb = th.empty(num, dim, dtype=th.float32, device=device)
        self.state_sum = self.emb.new().resize_(self.emb.size(0)).zero_()
        self.state_step = 0
        # queue used by asynchronous update
        self.async_q = None
        # asynchronous update process
        self.async_p = None

    def init(self, emb_init):
        """Initializing the embeddings.

        Parameters
        ----------
        emb_init : float
            The intial embedding range should be [-emb_init, emb_init].
        """
#         self.emb = self.emb.to("cpu")
        INIT.uniform_(self.emb, -emb_init, emb_init)
#         self.emb = self.emb.to(self.device)
        INIT.zeros_(self.state_sum)
        if self.basis is not None:
            self.basis.init(emb_init)
        
    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process tensor access
        """
        self.emb.share_memory_()
        self.state_sum.share_memory_()
        
        if self.basis is not None:
            self.basis.share_memory()

    def __call__(self, idx, gpu_id=-1, trace=True, mask=None):
        """ Return sliced tensor.

        Parameters
        ----------
        idx : th.tensor
            Slicing index
        gpu_id : int
            Which gpu to put sliced data in.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: True
        """
        s = self.emb[idx]
        if gpu_id >= 0:
            s = s.to('cuda:'+str(gpu_id))#cuda(gpu_id)
        # During the training, we need to trace the computation.
        # In this case, we need to record the computation path and compute the gradients.
        if trace:
            data = s.clone().detach().requires_grad_(True)
            self.trace.append((idx, data, mask))
        else:
            data = s
            
        if self.basis is not None:
#             print(data, self.basis(th.tensor([0], dtype=th.long)), self.basis(th.tensor([0], dtype=th.long)).reshape(-1, self.dim))
            data = th.einsum('ab,bc->ac', data, self.basis(th.tensor([0], dtype=th.long), gpu_id, trace).reshape(-1, self.dim))
#             print(data)
            
        return data

    def update(self, weight_decay = 0.0, norm_p = 3, gpu_id=-1, clr=None):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        if clr is None:
            clr = self.args.lr
        self.state_step += 1
        with th.no_grad():
            for idx, data, mask in self.trace:
                grad = data.grad.data
        
                device = self.state_sum.device
                if mask is not None:
                    grad_indices = idx[mask]
                    grad_values = grad[mask]
                    data_ = data[mask]
                else:
                    grad_indices = idx
                    grad_values = grad
                    data_ = data
                    
                output, inverse_indices = th.unique(grad_indices, sorted=True, return_inverse=True)
                unique_grad_values = th.zeros((len(output), grad_values.size(1)), dtype=th.float32, device=grad_values.device)
                
                # Get the indices of the unique elements
                perm = th.arange(inverse_indices.size(0), dtype=inverse_indices.dtype, device=inverse_indices.device)
                inverse_, perm = inverse_indices.flip([0]), perm.flip([0])
                perm = inverse_.new_empty(output.size(0)).scatter_(0, inverse_, perm)
                
                
                if grad_values.device != inverse_indices.device:
                    inverse_indices = inverse_indices.to(grad_values.device)
                unique_grad_values.index_add_(0, inverse_indices, grad_values)
                
                
                if weight_decay != 0 and norm_p > 0:
                    unique_grad_values = unique_grad_values + weight_decay*norm_p*th.pow(th.abs(data_[perm.to(data.device)]), norm_p-1)*th.sign(data_[perm.to(data.device)])
                grad_sum = (unique_grad_values * unique_grad_values).mean(1)
                
                if device != output.device:
                    output = output.to(device)
                if device != grad_sum.device:
                    grad_sum = grad_sum.to(device)
                self.state_sum.index_add_(0, output, grad_sum)
                std = self.state_sum[output]  # _sparse_mask
                if gpu_id >= 0:
                    std = std.cuda(gpu_id)
                std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                tmp = (-clr * unique_grad_values / std_values)
                if tmp.device != device:
                    tmp = tmp.to(device)
                self.emb.index_add_(0, output, tmp)
                
                
        self.trace = []
        
        if self.basis is not None:
            self.basis.update(weight_decay=0.0, norm_p=2, gpu_id=gpu_id, clr=self.args.basis_lr)

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
