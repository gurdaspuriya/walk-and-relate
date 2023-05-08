#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import os
import random
import torch
import dgl
import logging
import pickle
import time

from collections import defaultdict
from dataloader import EvalDataset, TrainDataset, TrainDataset_v2, NewBidirectionalOneShotIterator
from dataloader import get_dataset

from utils import get_compatible_batch_size, save_model, CommonArgParser

backend = os.environ.get('DGLBACKEND', 'pytorch')

import torch.multiprocessing as mp
from train_pytorch import load_model
from train_pytorch import train, train_mp
from train_pytorch import test, test_mp


# In[2]:


def set_seeds(seed=1234):
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = str(1)
    os.environ['MKL_NUM_THREADS'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = str(':16:8')
#     os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    


# In[3]:


# mpl = mp.log_to_stderr()
# mpl.setLevel(logging.DEBUG)


# In[4]:


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='A list of gpu ids, e.g. 0 1 2 4')
        self.add_argument('--mix_cpu_gpu', action='store_true',
                          help='Training a knowledge graph embedding model with both CPUs and GPUs.'\
                                  'The embeddings are stored in CPU memory and the training is performed in GPUs.'\
                                  'This is usually used for training a large knowledge graph embeddings.')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'\
                                  'The positive score will be adjusted '\
                                  'as pos_score = pos_score * edge_importance')
        self.add_argument('--walk_length', default=80, type=int, 
                            help="number of nodes in a sequence")
        self.add_argument('--window_size', default=5, type=int, 
                            help="context window size")
        self.add_argument('--num_workers', default=8, type=int, 
                            help="number of workers to create samples in the dataloader")
        self.add_argument('--composition_weights_file', type=str, 
                            help="Location of the file with importance weights and rules for the composed relations")
        self.add_argument('--composition_mapping', default='including', type=str, 
                            help="How to map compositions to pre-existing relations via rules: options are only (only use the compositions that can be mapped), including (default, map if rule available), none (no mapping) or ignore (no composition is used)")
        self.add_argument('--weight_sharing', default='no', type=str, 
                            help="How to share weights between the relation embeddings: no (default), model or rnn. model option is only valid for TransE or RotatE")
        self.add_argument('--num_bases', default='-1', type=int, 
                            help="Number of bases, if a common basis is used for weight sharing. Negative number if no basis sharing")
        self.add_argument('--basis_lr', default=0.0001, type=float, 
                            help="Learning rate for the basis vectors (used only when using basis)")
        self.add_argument('--rnn_lr', default=0.0001, type=float, 
                            help="Learning rate for the RNN (used only when using RNN)")
        self.add_argument('--early_stop_patience', default=2, type=int, 
                            help="Number of logging iteration to wait before early stoping if the validation metric does not improve.")
        self.add_argument('--early_stop_metric', default='MRR', type=str, 
                            help="Metric to observe for early stopping")
        self.add_argument('--seed', default=12345, type=int, 
                            help="Seed for random generators")
        self.add_argument('--sample_edges', default=0.5, type=float, 
                            help="Sampling probability of edges in the training data")


# In[5]:


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


# In[6]:


def get_compatible_neg_sample_size(batch_size, walk_length, window_size, neg_sample_size):
    num_pos = int(2 * walk_length * window_size - window_size * (window_size + 1))//2
    if (batch_size*num_pos) % neg_sample_size != 0:
        while True:
            neg_sample_size += 1
            if (batch_size*num_pos) % neg_sample_size == 0:
                print('batch_size, walk_length, window_wize and neg_sample_size are not compatible. Changed the neg_sample_size to {}'.format(
            neg_sample_size))
                break
    return neg_sample_size


# In[7]:


def get_composition_weights(composition_weights_file=None, composition_mapping='including'):
    
    # All relations are equally important if a relation importance file is not provided
    if composition_weights_file is None:
        return defaultdict(lambda: 1.0)
    
    # All compositions not in composition importance file are considered unimportant
    composition_confidence = defaultdict(lambda: 0.0)
    # The corresponding mapping to existing relations is empty
    composition_relation_mapping = defaultdict(lambda: [])
    
    # Load data (deserialize)
    with open(composition_weights_file, 'rb') as handle:
        temp = pickle.load(handle)
        for key in temp:
            composition_weight = temp[key][0]
            if len(key) == 1:
                composition_weight = 0.0
            if composition_mapping == 'ignore':
                if len(key) > 1:
                    composition_weight = 0.0
                    
            composition_dict = {}
            if composition_mapping == 'including' or composition_mapping == 'only' or composition_mapping == 'ignore' :
                composition_dict = temp[key][1]
            
            # What kind of compositions to keep
            if composition_mapping=='only':
                if len(composition_dict) == 0:
                    continue;
                
            
            composition_dict_reversed = {(_[0], -1*_[1]):composition_dict[_] for _ in composition_dict}
            key_reversed = tuple(list(reversed([(relation[0], relation[1]*-1) for relation in key])))
                
            composition_confidence[key_reversed] = composition_weight
            composition_relation_mapping[key_reversed] = list(composition_dict_reversed.items())
            composition_confidence[key] = composition_weight
            composition_relation_mapping[key] = list(composition_dict.items())
            
    return composition_confidence, composition_relation_mapping


# In[8]:


# def get_all_compositions_mapping(n_relations=1, max_composition_length=2, filtered_compositions=None):
#     import itertools
#     composition_to_id = {}
#     relation_expanded_list = [(r,1) for r in range(n_relations)] + [(r,-1) for r in range(n_relations)]
#     for r in range(n_relations):
#         composition_to_id[((r,1),)] = len(composition_to_id)
#     if filtered_compositions is not None:
#         for key in filtered_compositions:
#             if key not in composition_to_id:
#                 composition_to_id[key] = len(composition_to_id)
#     else:
#         for i in range(2, composition_length+1):
#             duplicated_relation_lists = [relation_expanded_list]*i
#             for key in itertools.product(*duplicated_relation_lists):
#                 key_reversed = tuple(list(reversed([(relation[0], relation[1]*-1) for relation in key])))
#                 if key < key_reversed:
#                     key = key_reversed
#                 if key not in composition_to_id:
#                     composition_to_id[key] = len(composition_to_id)
#     return composition_to_id


# In[9]:


# def get_all_compositions_mapping(n_relations=1, max_composition_length=2, filtered_compositions=None):
#     import itertools
#     composition_to_id = {}
#     relation_expanded_list = [(r,1) for r in range(n_relations)] + [(r,-1) for r in range(n_relations)]
#     for r in range(n_relations):
#         composition_to_id[((r,1),)] = list({(len(composition_to_id), 1):1.0}.items())
#     if filtered_compositions is not None:
#         for key in filtered_compositions:
#             if key not in composition_to_id:
#                 if len(filtered_compositions[key]) > 0:
#                     composition_to_id[key] = list(filtered_compositions[key].items())
#                 else:
#                     composition_to_id[key] = list({(len(composition_to_id), 1):1.0}.items())
#     else:
#         for i in range(2, composition_length+1):
#             duplicated_relation_lists = [relation_expanded_list]*i
#             for key in itertools.product(*duplicated_relation_lists):
#                 key_reversed = tuple(list(reversed([(relation[0], relation[1]*-1) for relation in key])))
#                 if key < key_reversed:
#                     key = key_reversed
#                 if key not in composition_to_id:
#                     composition_to_id[key] = list({(len(composition_to_id), 1):1.0}.items())
#     return composition_to_id


# In[10]:


# def get_all_compositions_mapping(n_relations=1, max_composition_length=2, filtered_compositions=None):
#     import itertools
#     composition_to_id = {}
#     relation_expanded_list = [(r,1) for r in range(n_relations)] + [(r,-1) for r in range(n_relations)]
#     for r in range(n_relations):
#         composition_to_id[((r,1),)] = len(composition_to_id)
#     if filtered_compositions is not None:
#         for key in filtered_compositions:
#             if key not in composition_to_id:
#                 composition_to_id[key] = len(composition_to_id)
#     else:
#         for i in range(2, composition_length+1):
#             duplicated_relation_lists = [relation_expanded_list]*i
#             for key in itertools.product(*duplicated_relation_lists):
#                 key_reversed = tuple(list(reversed([(relation[0], relation[1]*-1) for relation in key])))
#                 if key < key_reversed:
#                     key = key_reversed
#                 if key not in composition_to_id:
#                     composition_to_id[key] = len(composition_to_id)
#     return composition_to_id


# In[11]:


def get_all_compositions_mapping(n_relations=1, max_composition_length=2, filtered_compositions=None):
    import itertools
    composition_to_id = {}
    relation_expanded_list = [(r,1) for r in range(n_relations)] + [(r,-1) for r in range(n_relations)]
    for r in range(n_relations):
        composition_to_id[((r,1),)] = list({(len(composition_to_id), 1):1.0}.items())
    if filtered_compositions is not None:
        for key in filtered_compositions:
            if key not in composition_to_id:
                key_reversed = tuple(list(reversed([(relation[0], relation[1]*-1) for relation in key])))
                if len(filtered_compositions[key]) > 0:
                    composition_dict_reversed = [((_[0][0], -1*_[0][1]), _[1]) for _ in filtered_compositions[key]]
                    composition_to_id[key] = filtered_compositions[key]
                    composition_to_id[key_reversed] = composition_dict_reversed
                else:
                    if key < key_reversed:
                        composition_to_id[key] = list({(len(composition_to_id), 1):1.0}.items())
                        composition_to_id[key_reversed] = list({(len(composition_to_id), -1):1.0}.items())
                    else:
                        composition_to_id[key] = list({(len(composition_to_id), -1):1.0}.items())
                        composition_to_id[key_reversed] = list({(len(composition_to_id), 1):1.0}.items())
                    
    else:
        for i in range(2, composition_length+1):
            duplicated_relation_lists = [relation_expanded_list]*i
            for key in itertools.product(*duplicated_relation_lists):
                key_reversed = tuple(list(reversed([(relation[0], relation[1]*-1) for relation in key])))
                if key not in composition_to_id:
                    if key < key_reversed:
                        composition_to_id[key] = list({(len(composition_to_id), 1):1.0}.items())
                        composition_to_id[key_reversed] = list({(len(composition_to_id), -1):1.0}.items())
                    else:
                        composition_to_id[key] = list({(len(composition_to_id), -1):1.0}.items())
                        composition_to_id[key_reversed] = list({(len(composition_to_id), 1):1.0}.items())
    return composition_to_id


# In[12]:


def main():
    args = ArgParser().parse_args()
    prepare_save_path(args)
    
    if args.num_proc == 1 and args.num_workers <= 1:
        set_seeds(args.seed);
    if args.weight_sharing == 'model':
        assert (args.model_name=='TransE' or args.model_name=='TransE_l2' or args.model_name=='TransE_l1' or args.model_name=='RotatE'),         'weight_sharing=model is only compatible with models TransE or RotatE'
    
    
    # Make sure that the positive edges and negative edges are feasible
    args.neg_sample_size = get_compatible_neg_sample_size(args.batch_size, args.walk_length, args.window_size, args.neg_sample_size)
    init_time_start = time.time()
    # load dataset and samplers
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)
    
    if args.sample_edges <= 0.99999:
        selected_train_indices = np.random.choice(len(dataset.train[0]), int(len(dataset.train[0])*args.sample_edges), replace=False)
        dataset.train = [dataset.train[i][selected_train_indices] for i in range(len(dataset.train))]
        unique, counts = np.unique(np.concatenate([dataset.train[0], dataset.train[2]]), return_counts=True)
        unique_rels = np.unique(dataset.train[1])
        print(len(unique), len(unique_rels), len(np.unique(dataset.valid[1])), len(np.unique(dataset.test[1])))
        selected_nids = [unique[i] for i in range(len(unique))]
        selected_valid_index = np.array([i for i in range(len(dataset.valid[0])) if ((dataset.valid[0][i] in selected_nids) and (dataset.valid[2][i] in selected_nids) and (dataset.valid[1][i] in unique_rels))])
        dataset.valid = (dataset.valid[0][selected_valid_index], dataset.valid[1][selected_valid_index], dataset.valid[2][selected_valid_index])
        selected_test_index = np.array([i for i in range(len(dataset.test[0])) if ((dataset.test[0][i] in selected_nids) and (dataset.test[2][i] in selected_nids) and (dataset.test[1][i] in unique_rels))])
        dataset.test = (dataset.test[0][selected_test_index], dataset.test[1][selected_test_index], dataset.test[2][selected_test_index])
    
    train_triples = set(list(zip(dataset.train[0], dataset.train[1], dataset.train[2])))
    valid_triples = set(list(zip(dataset.valid[0], dataset.valid[1], dataset.valid[2])))
    test_triples = set(list(zip(dataset.test[0], dataset.test[1], dataset.test[2])))
    
    print('Valid triples common with train triples: ', len(train_triples & valid_triples))
    print('Test triples common with train triples: ', len(train_triples & test_triples))
    
    

    if args.neg_sample_size_eval < 0:
        args.neg_sample_size_eval = dataset.n_entities
    args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)
    
    # Load the relation confidence map
    composition_confidence, rule_mapping = get_composition_weights(args.composition_weights_file, args.composition_mapping)
    # Lets map the compositions to integer IDs
    composition_to_id_mapping = None
    if args.weight_sharing != 'model' and args.weight_sharing != 'rnn':
        if args.composition_weights_file is None:
            # All possible compositions
            composition_to_id_mapping = get_all_compositions_mapping(n_relations=dataset.n_relations, max_composition_length=args.window_size, filtered_compositions=None)
        else:
            composition_to_id_mapping = get_all_compositions_mapping(n_relations=dataset.n_relations, max_composition_length=args.window_size, filtered_compositions=rule_mapping)
    
    # We should turn on mix CPU-GPU training for multi-GPU training.
    if len(args.gpu) > 1:
        args.mix_cpu_gpu = True
        if args.num_proc < len(args.gpu):
            args.num_proc = len(args.gpu)
    # We need to ensure that the number of processes should match the number of GPUs.
    if len(args.gpu) > 1 and args.num_proc > 1:
        assert args.num_proc % len(args.gpu) == 0,                 'The number of processes needs to be divisible by the number of GPUs'
    # For multiprocessing training, we need to ensure that training processes are synchronized periodically.
    if args.num_proc > 1 and args.force_sync_interval is None:
        args.force_sync_interval = 1000

    args.eval_filter = not args.no_eval_filter
    if args.neg_deg_sample_eval:
        assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

    train_data = TrainDataset_v2(dataset, args, ranks=args.num_proc, has_importance=args.has_edge_importance)
    

    if args.num_proc > 1:
        train_samplers = []
        for i in range(args.num_proc):
            # for each GPU, allocate num_proc // num_GPU processes
            train_sampler_random_walk = train_data.create_random_walk_sampler(batch_size=args.batch_size,
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           rank=i)
            
            train_sampler_head = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='head',
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                           args.neg_sample_size,
                                                           args.neg_sample_size,
                                                           mode='tail',
                                                           num_workers=args.num_workers,
                                                           shuffle=True,
                                                           exclude_positive=False,
                                                           rank=i)
            train_samplers.append(NewBidirectionalOneShotIterator(train_sampler_random_walk, train_sampler_head, train_sampler_tail,
                                                                  args.neg_sample_size, args.neg_sample_size,
                                                                  True, dataset.n_entities,args.walk_length, args.window_size, 
                                                        args.batch_size, composition_confidence, rule_mapping, composition_to_id_mapping, args.has_edge_importance, train_data.g))

        train_sampler = NewBidirectionalOneShotIterator(train_sampler_random_walk, train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                       True, dataset.n_entities, args.walk_length, args.window_size, 
                                                        args.batch_size, composition_confidence, rule_mapping, composition_to_id_mapping, args.has_edge_importance, train_data.g)
    else: # This is used for debug
        
        train_sampler_random_walk = train_data.create_random_walk_sampler(args.batch_size,
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       rank=0)
        train_sampler_head = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='head',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler_tail = train_data.create_sampler(args.batch_size,
                                                       args.neg_sample_size,
                                                       args.neg_sample_size,
                                                       mode='tail',
                                                       num_workers=args.num_workers,
                                                       shuffle=True,
                                                       exclude_positive=False)
        train_sampler = NewBidirectionalOneShotIterator(train_sampler_random_walk, train_sampler_head, train_sampler_tail,
                                                        args.neg_sample_size, args.neg_sample_size,
                                                        True, dataset.n_entities, args.walk_length, args.window_size, 
                                                        args.batch_size, composition_confidence, rule_mapping, composition_to_id_mapping, args.has_edge_importance, train_data.g)
    
    if args.valid or args.test:
        if len(args.gpu) > 1:
            args.num_test_proc = args.num_proc if args.num_proc < len(args.gpu) else len(args.gpu)
        else:
            args.num_test_proc = args.num_proc
        if args.valid:
            assert dataset.valid is not None, 'validation set is not provided'
        if args.test:
            assert dataset.test is not None, 'test set is not provided'
        eval_dataset = EvalDataset(dataset, args)

    if args.valid:
        if args.num_proc > 1:
            valid_sampler_heads = []
            valid_sampler_tails = []
            if args.dataset == "wikikg90M":
            	for i in range(args.num_proc):
	                valid_sampler_tail = eval_dataset.create_sampler_wikikg90M('valid', args.batch_size_eval,
	                                                                  mode='tail',
	                                                                  rank=i, ranks=args.num_proc)
	                valid_sampler_tails.append(valid_sampler_tail)
            else:
	            for i in range(args.num_proc):
	                valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
	                                                                  args.neg_sample_size_eval,
	                                                                  args.neg_sample_size_eval,
	                                                                  args.eval_filter,
	                                                                  mode='chunk-head',
	                                                                  num_workers=args.num_workers,
	                                                                  rank=i, ranks=args.num_proc)
	                valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
	                                                                  args.neg_sample_size_eval,
	                                                                  args.neg_sample_size_eval,
	                                                                  args.eval_filter,
	                                                                  mode='chunk-tail',
	                                                                  num_workers=args.num_workers,
	                                                                  rank=i, ranks=args.num_proc)
	                valid_sampler_heads.append(valid_sampler_head)
	                valid_sampler_tails.append(valid_sampler_tail)
        else: # This is used for debug
            if args.dataset == "wikikg90M":
                valid_sampler_tail = eval_dataset.create_sampler_wikikg90M('valid', args.batch_size_eval,
                                                             mode='tail',
                                                             rank=0, ranks=1)
            else:
                valid_sampler_head = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.eval_filter,
                                                                 mode='chunk-head',
                                                                 num_workers=args.num_workers,
                                                                 rank=0, ranks=1)
                valid_sampler_tail = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.neg_sample_size_eval,
                                                                 args.eval_filter,
                                                                 mode='chunk-tail',
                                                                 num_workers=args.num_workers,
                                                                 rank=0, ranks=1)
    if args.test:
        if args.num_test_proc > 1:
            test_sampler_tails = []
            test_sampler_heads = []
            if args.dataset == "wikikg90M":
            	for i in range(args.num_proc):
	                valid_sampler_tail = eval_dataset.create_sampler_wikikg90M('test', args.batch_size_eval,
	                                                                  mode='tail',
	                                                                  rank=i, ranks=args.num_proc)
	                valid_sampler_tails.append(valid_sampler_tail)
            else:
	            for i in range(args.num_test_proc):
	                test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
	                                                                 args.neg_sample_size_eval,
	                                                                 args.neg_sample_size_eval,
	                                                                 args.eval_filter,
	                                                                 mode='chunk-head',
	                                                                 num_workers=args.num_workers,
	                                                                 rank=i, ranks=args.num_test_proc)
	                test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
	                                                                 args.neg_sample_size_eval,
	                                                                 args.neg_sample_size_eval,
	                                                                 args.eval_filter,
	                                                                 mode='chunk-tail',
	                                                                 num_workers=args.num_workers,
	                                                                 rank=i, ranks=args.num_test_proc)
	                test_sampler_heads.append(test_sampler_head)
                	test_sampler_tails.append(test_sampler_tail)
        else:
        	if args.dataset == "wikikg90M":
        		test_sampler_tail = eval_dataset.create_sampler_wikikg90M('test', args.batch_size_eval,
                                                            mode='tail',
                                                            rank=0, ranks=1)
        	else:
	            test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
	                                                            args.neg_sample_size_eval,
	                                                            args.neg_sample_size_eval,
	                                                            args.eval_filter,
	                                                            mode='chunk-head',
	                                                            num_workers=args.num_workers,
	                                                            rank=0, ranks=1)
	            test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
	                                                            args.neg_sample_size_eval,
	                                                            args.neg_sample_size_eval,
	                                                            args.eval_filter,
	                                                            mode='chunk-tail',
	                                                            num_workers=args.num_workers,
	                                                            rank=0, ranks=1)

    # load model
    
    # Weight sharing is there
    if composition_to_id_mapping is None:
        model = load_model(args, dataset.n_entities, dataset.n_relations)
    else:
        model = load_model(args, dataset.n_entities, len(composition_to_id_mapping))
        
    if args.num_proc > 1:
        model.share_memory()

    emap_file = dataset.emap_fname
    rmap_file = dataset.rmap_fname
    # We need to free all memory referenced by dataset.
    eval_dataset = None
    dataset = None

    print('Total initialize time {:.3f} seconds'.format(time.time() - init_time_start))

    # train
    start = time.time()
    if args.num_proc > 1:
        procs = []
        barrier = mp.Barrier(args.num_proc)
        valid_queue = mp.Queue(args.num_proc)if args.valid else None
        for i in range(args.num_proc):
            if args.dataset == "wikikg90M":
                valid_sampler = [valid_sampler_tails[i]] if args.valid else None
            else:
                valid_sampler = [valid_sampler_heads[i], valid_sampler_tails[i]] if args.valid else None
            proc = mp.Process(target=train_mp, args=(args,
                                                     model,
                                                     train_samplers[i],
                                                     valid_sampler,
                                                     i,
                                                     barrier,
                                                     valid_queue))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
    else:
        if args.dataset == "wikikg90M":
            valid_samplers = [valid_sampler_tail] if args.valid else None
        else:
            valid_samplers = [valid_sampler_head, valid_sampler_tail] if args.valid else None
        train(args, model, train_sampler, valid_samplers)

    print('training takes {} seconds'.format(time.time() - start))

    if not args.no_save_emb:
        save_model(args, model, emap_file, rmap_file)
    # test
    if args.test:
        start = time.time()
        if args.num_test_proc > 1:
            queue = mp.Queue(args.num_test_proc)
#             queue = None
            procs = []
            for i in range(args.num_test_proc):
                if args.dataset == "wikikg90M":
                    proc = mp.Process(target=test_mp, args=(args,
                                                            model,
                                                            [test_sampler_tails[i]],
                                                            i,
                                                            'Test',
                                                            queue))
                else:
                    proc = mp.Process(target=test_mp, args=(args,
                                                            model,
                                                            [test_sampler_heads[i], test_sampler_tails[i]],
                                                            i,
                                                            'Test',
                                                            queue))
                procs.append(proc)
                proc.start()

            if args.dataset == "wikikg90M":
                print('The predict results have saved to {}'.format(args.save_path))
            else:
                total_metrics = {}
                metrics = {}
                logs = []
                for i in range(args.num_test_proc):
                    log = queue.get()
                    logs = logs + log

                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
                print("-------------- Test result --------------")
                for k, v in metrics.items():
                    print('Test average {} : {}'.format(k, v))
                print("-----------------------------------------")

            for proc in procs:
                proc.join()
        else:
            if args.dataset == "wikikg90M":
                test(args, model, [test_sampler_tail])
            else:
                test(args, model, [test_sampler_head, test_sampler_tail])
            if args.dataset == "wikikg90M":
                print('The predict results have saved to {}'.format(args.save_path))
        print('testing takes {:.3f} seconds'.format(time.time() - start))


# In[ ]:


if __name__ == '__main__':
    main()