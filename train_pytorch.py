# -*- coding: utf-8 -*-
#
# train_pytorch.py
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

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
from models.pytorch.tensor_models import thread_wrapped_func
from models import KEModel
from utils import save_model, get_compatible_batch_size

import os
import logging
import time
from functools import wraps

import dgl
import dgl.backend as F

from dataloader import EvalDataset
from dataloader import get_dataset

def load_model(args, n_entities, n_relations, composition_to_id_mapping=None, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model

def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path):
    model = load_model(args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, valid_samplers=None, rank=0, barrier=None, valid_queue=None):
    # Early stopping criteris
    best_metric = -1
    checks_no_improve = 0
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    for step in range(0, args.max_step):
#         if step >= 2000 and step %2000 == 0:
#             args.rnn_lr /= 2
        start1 = time.time()
        walk_g, pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1
        start1 = time.time()
        loss, log = model.forward(walk_g, pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        
        model.update(gpu_id)
        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
            (step + 1) % args.force_sync_interval == 0:
            if barrier is not None:
                barrier.wait()

        if (step + 1) % args.log_interval == 0:
            for k in logs[0].keys():
                v = sum(l[k] for l in logs) / len(logs)
                print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
            logs = []
            print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                            time.time() - start))
            print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                rank, sample_time, forward_time, backward_time, update_time))
            sample_time = 0
            update_time = 0
            forward_time = 0
            backward_time = 0
            start = time.time()
           

        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            results_dict = {}
            results_dict = test(args, model, valid_samplers, rank, mode='Valid', queue=valid_queue)
            print('[proc {}]validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if valid_queue is not None:
                if rank == 0:
                    total_metrics = {}
                    metrics = {}
                    validation_logs = []
                    for i in range(args.num_proc):
                        log = valid_queue.get()
                        validation_logs = logs + log

                    for metric in validation_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in validation_logs]) / len(validation_logs)
                    print("-------------- Validation result --------------")
                    for k, v in metrics.items():
                        results_dict[k] = v
                        print('Valid average {} : {}'.format(k, v))
                    print("-----------------------------------------")
                    for _ in range(args.num_proc):
                        valid_queue.put(results_dict)
                
            if barrier is not None:
                barrier.wait()
            if valid_queue is not None:
                results_dict = valid_queue.get()
                
                
            # Check if validation metric increases
            if args.early_stop_metric == 'MR':
                validation_metric = -results_dict[args.early_stop_metric]
            else:
                validation_metric = results_dict[args.early_stop_metric]
            if validation_metric - best_metric >= 1e-6:
                # Track improvement
                checks_no_improve = 0
                best_metric = validation_metric

            # Otherwise increment count of checks with no improvement
            else:
                checks_no_improve += 1
                # Trigger early stopping
                if checks_no_improve >= args.early_stop_patience:
                    print(
                        f'\nEarly Stopping! Total iterations: {step+1}.'
                    )
                    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
                    return
                
            

    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))

def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1
        
    results_dict = {}

    if args.dataset == "wikikg90M":
        with th.no_grad():
            logs = []
            answers = []
            for sampler in test_samplers:
                for query, ans, candidate in sampler:
                    model.forward_test_wikikg(query, ans, candidate, mode, logs, gpu_id)
                    answers.append(ans)
            print("[{}] finished {} forward".format(rank, mode))

            for i in range(len(test_samplers)):
                test_samplers[i] = test_samplers[i].reset()

            if mode == "Valid":
                metrics = {}
                if len(logs) > 0:
                    for metric in logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                if queue is not None:
                    queue.put(logs)
                else:
                    for k, v in metrics.items():
                        print('[{}]{} average {}: {}'.format(rank, mode, k, v))
            else:
                input_dict = {}
                input_dict['h,r->t'] = {'t_correct_index': th.cat(answers, 0), 't_pred_top10': th.cat(logs, 0)}
                th.save(input_dict, os.path.join(args.save_path, "test_{}.pkl".format(rank)))
    else:
        with th.no_grad():
            logs = []

            for i, sampler in enumerate(test_samplers):
                for pos_g, neg_g in sampler:
                    model.forward_test(pos_g, neg_g, logs, gpu_id)
            metrics = {}
            if len(logs) > 0:
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
            if queue is not None:
                queue.put(logs)
            else:
                for k, v in metrics.items():
                    results_dict[k] = v
                    print('[{}]{} average {}: {}'.format(rank, mode, k, v))
        test_samplers[0] = test_samplers[0].reset()
        test_samplers[1] = test_samplers[1].reset()
        
    return results_dict

@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, rank=0, barrier=None, valid_queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers, rank, barrier, valid_queue)

@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, rank, mode, queue)