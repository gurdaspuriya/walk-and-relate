#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Read https://bugs.python.org/issue17560
# Multiprocessing has issues with pickling large objects as it uses protocol 3 in python 3.6
# We patch the pickling functionality as described in https://stackoverflow.com/questions/47776486/python-struct-error-i-format-requires-2147483648-number-2147483647
import functools
import logging
import struct
import sys
import traceback

logger = logging.getLogger()


def patch_mp_connection_bpo_17560():
    """Apply PR-10305 / bpo-17560 connection send/receive max size update

    See the original issue at https://bugs.python.org/issue17560 and 
    https://github.com/python/cpython/pull/10305 for the pull request.

    This only supports Python versions 3.3 - 3.7, this function
    does nothing for Python versions outside of that range.

    """
    patchname = "Multiprocessing connection patch for bpo-17560"
    if not (3, 3) < sys.version_info < (3, 8):
        logger.info(
            patchname + " not applied, not an applicable Python version: %s",
            sys.version
        )
        return

    from multiprocessing.connection import Connection

    orig_send_bytes = Connection._send_bytes
    orig_recv_bytes = Connection._recv_bytes

    @functools.wraps(orig_send_bytes)
    def send_bytes(self, buf):
        n = len(buf)
        if n > 0x7fffffff:
            pre_header = struct.pack("!i", -1)
            header = struct.pack("!Q", n)
            self._send(pre_header)
            self._send(header)
            self._send(buf)
        else:
            orig_send_bytes(self, buf)

    @functools.wraps(orig_recv_bytes)
    def recv_bytes(self, maxsize=None):
        buf = self._recv(4)
        size, = struct.unpack("!i", buf.getvalue())
        if size == -1:
            buf = self._recv(8)
            size, = struct.unpack("!Q", buf.getvalue())
        if maxsize is not None and size > maxsize:
            return None
        return self._recv(size)

    Connection._send_bytes = send_bytes
    Connection._recv_bytes = recv_bytes

    logger.info(patchname + " applied")


# In[2]:


patch_mp_connection_bpo_17560()
import multiprocessing as mp


# In[3]:


import argparse
import os
import logging
import time
import torch
import dgl
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from dataloader import get_dataset
import random
from scipy.optimize import minimize_scalar
backend = os.environ.get('DGLBACKEND', 'pytorch')


# In[4]:


EPSILON=0.00001


# In[5]:


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--data_path', type=str, default='data',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--save_path', type=str, default='composed_relations.pkl',
                          help='The path of the pkl file to write the results.')
        self.add_argument('--dataset', default='FB15k', type=str, 
                            help="dataset")
        self.add_argument('--log_interval', default=10, type=int, 
                            help="log interval in seconds")
        self.add_argument('--max_level', default=3, type=int, 
                            help="max length of composition: based on context window size")
        self.add_argument('--support', default=0.1, type=float, 
                            help="minimum support threshold to consider a composition as significant")
        self.add_argument('--confidence', default=0.5, type=float, 
                            help="minimum confidence threshold to consider a composition -> relation as significant")
        self.add_argument('--num_workers', default=8, type=int, 
                            help="number of workers to create samples in the dataloader")
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs, '\
                                  'the foramt should be built_in. For users own knowledge graphs, '\
                                  'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--delimiter', type=str, default='\t',
                          help='Delimiter used in data files. Note all files should use the same delimiter.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE '\
                                  'on their own datasets. If the format is raw_udd_{htr}, '\
                                  'users need to provide train_file [valid_file] [test_file]. '\
                                  'If the format is udd_{htr}, users need to provide '\
                                  'entity_file relation_file train_file [valid_file] [test_file]. '\
                                  'In both cases, valid_file and test_file are optional.')
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'\
                                  'The positive score will be adjusted '\
                                  'as pos_score = pos_score * edge_importance')
        self.add_argument('--sampling_prob', default=1.0, type=float,
                          help='Use a fraction of the edges to estimate the statistics, useful for large graphs')
        self.add_argument('--add_correction', action='store_true',
                          help='Add correction factor for sampled graphs')
        self.add_argument('--join_type', default='pre', type=str,
                          help='How to do the join, if "pre", we do join then groupby to create children nodes. \
                          This can be memory intensive, but fast. if "post", we do individual joins for each children \
                          node. This takes less memory but more time.')


# In[6]:


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


# In[7]:


def ConstructGraph(edges, n_entities, args):
    """Construct Graph to count metapaths

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list
    n_entities : int
        number of entities
    args :
        Global configs.
    """
    num_edges = len(edges[0])
    random.seed(1234)
    
    sampled_eids = np.array(sorted(random.sample(list(range(num_edges)), int(num_edges*args.sampling_prob))))
    
    
    if args.has_edge_importance:
        src, etype_id, dst, e_impts = edges
        src = src[sampled_eids]
        etype_id = etype_id[sampled_eids]
        dst = dst[sampled_eids]
    else:
        src, etype_id, dst = edges
        src = src[sampled_eids]
        etype_id = etype_id[sampled_eids]
        dst = dst[sampled_eids]
        
    
        
    etype_direction = torch.cat((torch.tensor([1]*len(etype_id)), torch.tensor([-1]*len(etype_id))) ,dim=0)
    etype_id = torch.cat((torch.tensor(etype_id), torch.tensor(etype_id)) ,dim=0)
       
    g = dgl.graph((torch.cat((torch.tensor(src), torch.tensor(dst)) ,dim=0), torch.cat((torch.tensor(dst), torch.tensor(src)) ,dim=0)))
    g.edata['tid'] = etype_id
    g.edata['direction'] = etype_direction
    return g


# In[8]:


def sample_correction_estimate_fn(x, sample_zeros, sample_edges, sample_probability, metapath_length, num_sampled_metapaths):
    """Correction factor for sampled graphs
    Returns the minimization function that models number of estimated edges in the original graph 
    which are a part of at least one compositional relation
    """
    num_upsampled_metapaths = num_sampled_metapaths/pow(sample_probability, metapath_length)
    fn = -sample_zeros +     (sample_edges - sample_probability*x) +     sample_probability*x*    pow((1 - pow(sample_probability, metapath_length-1)),         (num_upsampled_metapaths)/x)
    return fn**2


# In[9]:


def sample_correction_estimate_pr(x, sample_zeros, sample_edges, sample_probability, metapath_length, num_sampled_metapaths):
    """Correction factor for sampled graphs
    Returns the minimization function that models number of estimated edges in the original graph 
    which are a part of at least one compositional relation
    """
    num_upsampled_metapaths = num_sampled_metapaths/pow(sample_probability, metapath_length)
    fn = -sample_zeros +     (sample_edges - sample_probability*x) +     sample_probability*x*    pow((1 - pow(sample_probability, metapath_length-1)),         (num_upsampled_metapaths)/x)
    print(sample_zeros, sample_edges, sample_probability, metapath_length, num_sampled_metapaths)
    print(-sample_zeros,     (sample_edges - sample_probability*x),           (1 - pow(sample_probability, metapath_length-1),            pow((1 - pow(sample_probability, metapath_length-1)),         (num_upsampled_metapaths)/x)),          num_upsampled_metapaths/x, num_upsampled_metapaths,     sample_probability*x*    pow((1 - pow(sample_probability, metapath_length-1)),         (num_upsampled_metapaths)/x))
    return fn


# In[10]:


def sample_correction_estimate(sample_zeros, sample_edges, sample_probability, metapath_length, num_sampled_metapaths):
    """Correction factor for sampled graphs
    Returns the number of estimated edges in the original graph 
    which are a part of at least one compositional relation
    """
    lower_bound = max(1, (sample_edges-sample_zeros)/sample_probability)
    upper_bound = min(sample_edges/sample_probability, num_sampled_metapaths/pow(sample_probability, metapath_length))
    res = minimize_scalar(lambda x: sample_correction_estimate_fn(x, sample_zeros, sample_edges, sample_probability, metapath_length, num_sampled_metapaths), bounds=(lower_bound, upper_bound), method='bounded')
#     sample_correction_estimate_pr(res.x, sample_zeros, sample_edges, sample_probability, metapath_length, num_sampled_metapaths)
    
    return res.x*sample_probability/sample_edges



# In[11]:


def populate_worker_pre_join(task_queue, output_queue, relation_counts, max_level=2, support=0.1, confidence=0.5, sampling_prob=1.0, add_correction=True, graph_df=None):
    while True:
        trie_node = task_queue.get()         # Read from the queue and do nothing
        
        # max length of the composed relations
        if trie_node.level == max_level:
            task_queue.task_done()
            continue
        rsuffix = '_'+str(trie_node.level)
        prev_rsuffix = '_'+str(trie_node.level-1)
        parent_df = trie_node.filtered_df
        
        # Joining requires the two columns to be index, while merge does not, join seems to be faster, 
        # but sometimes give value-error and maked dst as index. Thus, using merge
#         all_children_df = trie_node.filtered_df.set_index('dst').join(graph_df.set_index('src')[['dst', 'tid', 'direction', 'eid']], rsuffix=rsuffix, how='inner').rename(columns={"tid": "tid"+rsuffix, "direction": "direction"+rsuffix, "eid": "eid"+rsuffix})
        # The following performs merge, which seems slower than setting index then joining
        all_children_df = trie_node.filtered_df.merge(graph_df[['src', 'dst', 'tid', 'direction', 'eid']], left_on='dst', right_on='src', suffixes=('', rsuffix)).drop('dst', 1).rename(columns={"tid": "tid"+rsuffix, "direction": "direction"+rsuffix, "eid": "eid"+rsuffix, "dst"+rsuffix: "dst"})
        all_children_df = all_children_df[~((all_children_df['tid'+prev_rsuffix] == all_children_df['tid'+rsuffix]) & (all_children_df['direction'+prev_rsuffix]*all_children_df['direction'+rsuffix] == -1))]
        temp = all_children_df.groupby(['tid'+rsuffix, 'direction'+rsuffix])
        for index, df in all_children_df.groupby(['tid'+rsuffix, 'direction'+rsuffix]):
            rule_rhs = pd.merge(df[['src', 'dst']], graph_df[['src', 'dst', 'tid', 'direction']],  how='inner', on=['src','dst'])
            val_counts_rhs = rule_rhs.value_counts(['tid', 'direction'])
            rhs_relation_mappings = {}
            for key in val_counts_rhs.index:
                if val_counts_rhs[key]/relation_counts[key] < support:
                    continue;
                rhs_confidence = val_counts_rhs[key]/len(df)
                if rhs_confidence < confidence:
                    continue;
                rhs_relation_mappings[key] = rhs_confidence    
#             rhs_relation = None
#             rhs_confidence = 1.0
#             if len(val_counts_rhs) > 0:
#                 rhs_relation = val_counts_rhs.idxmax()
#                 rhs_relation_cnt = val_counts_rhs[rhs_relation]
# #                 if rhs_relation_cnt/relation_counts[rhs_relation] < support:
# #                     rhs_relation = None
#                 rhs_confidence = rhs_relation_cnt/len(df)
# #                 if rhs_confidence < confidence:
# #                     rhs_relation = None
# #                     rhs_confidence = 1.0
#             if rhs_relation is not None:
#                 print(rhs_relation)
            path_weight = 1.0
            break_flag = False
            for i in range(trie_node.level):
                unique_eids = df['eid'+'_'+str(i)].unique()
                if (sampling_prob < (1.0 - EPSILON)) and add_correction:
                    path_weight *= sample_correction_estimate(relation_counts[trie_node.path[i]]-len(unique_eids), relation_counts[trie_node.path[i]], sampling_prob, trie_node.level+1, len(df.index))
                else:
                    path_weight *= (len(unique_eids)/relation_counts[trie_node.path[i]]);
                if path_weight <= support:
                    break_flag = True
                    break;
            if break_flag:
                continue;
            unique_eids = df['eid'+rsuffix].unique()
            if (sampling_prob < (1.0 - EPSILON)) and add_correction:
                path_weight *= sample_correction_estimate(relation_counts[index]-len(unique_eids), relation_counts[index], sampling_prob, trie_node.level+1, len(df.index))
            else:
                path_weight *= (len(unique_eids)/relation_counts[index]);
                
            if path_weight <= support:
                continue;
            trie_node_child = TrieNode(label=index, level=trie_node.level+1, path=trie_node.path+[index], path_weight=path_weight, filtered_df=df)
            task_queue.put(trie_node_child)
            output_queue.put([trie_node.path+[index], path_weight, rhs_relation_mappings])
        task_queue.task_done()


# In[12]:


def populate_worker_post_join(task_queue, output_queue, relation_counts, max_level=2, support=0.1, confidence=0.5, sampling_prob=1.0, add_correction=True, graph_df=None):
    while True:
        trie_node = task_queue.get()         # Read from the queue and do nothing
        
        # max length of the composed relations
        if trie_node.level == max_level:
            task_queue.task_done()
            continue
        rsuffix = '_'+str(trie_node.level)
        prev_rsuffix = '_'+str(trie_node.level-1)
        parent_df = trie_node.filtered_df
        
        for relation in relation_counts:
            temp_df = graph_df[(graph_df['tid'] == relation[0]) & (graph_df['direction'] == relation[1])]
            
            # Joining requires the two columns to be index, while merge does not, join seems to be faster, 
            # but sometimes give value-error and maked dst as index. Thus, using merge
#             children_df = trie_node.filtered_df.set_index('dst').join(temp_df.set_index('src')[['dst', 'tid', 'direction', 'eid']], rsuffix=rsuffix, how='inner').rename(columns={"tid": "tid"+rsuffix, "direction": "direction"+rsuffix, "eid": "eid"+rsuffix})
            children_df = trie_node.filtered_df.merge(temp_df[['src', 'dst', 'tid', 'direction', 'eid']], left_on='dst', right_on='src', suffixes=('', rsuffix)).drop('dst', 1).rename(columns={"tid": "tid"+rsuffix, "direction": "direction"+rsuffix, "eid": "eid"+rsuffix, "dst"+rsuffix: "dst"})
            
            children_df = children_df[~((children_df['tid'+prev_rsuffix] == children_df['tid'+rsuffix]) & (children_df['direction'+prev_rsuffix]*children_df['direction'+rsuffix] == -1))]
            if len(children_df) == 0:
                continue
#             try:
#                 rule_rhs = pd.merge(children_df[['src', 'dst']], graph_df[['src', 'dst', 'tid', 'direction']],  how='inner', on=['src','dst'])
#             except Exception as e:
#                 print(children_df.index.name, graph_df.index.name)
#                 print('Caught exception in worker thread (x = %d):' % 0)

#                 # This prints the type, value, and stack trace of the
#                 # current exception being handled.
#                 traceback.print_exc()

#                 raise e
            rule_rhs = pd.merge(children_df[['src', 'dst']], graph_df[['src', 'dst', 'tid', 'direction']],  how='inner', on=['src','dst'])
#             print(rule_rhs)
            val_counts_rhs = rule_rhs.value_counts(['tid', 'direction'])
            rhs_relation_mappings = {}
            for key in val_counts_rhs.index:
                if val_counts_rhs[key]/relation_counts[key] < support:
                    continue;
                rhs_confidence = val_counts_rhs[key]/len(children_df.index)
                if rhs_confidence < confidence:
                    continue;
                rhs_relation_mappings[key] = rhs_confidence   
            
            path_weight = 1.0
            break_flag = False
            
            for i in range(trie_node.level):
                unique_eids = children_df['eid'+'_'+str(i)].unique()
                if (sampling_prob < (1.0 - EPSILON)) and add_correction:
                    path_weight *= sample_correction_estimate(relation_counts[trie_node.path[i]]-len(unique_eids), relation_counts[trie_node.path[i]], sampling_prob, trie_node.level+1, len(children_df.index))
                else:
                    path_weight *= (len(unique_eids)/relation_counts[trie_node.path[i]]);
                if path_weight <= support:
                    break_flag = True
                    break;
            if break_flag:
                continue;
            unique_eids = children_df['eid'+rsuffix].unique()
            if (sampling_prob < (1.0 - EPSILON)) and add_correction:
                path_weight *= sample_correction_estimate(relation_counts[relation]-len(unique_eids), relation_counts[relation], sampling_prob, trie_node.level+1, len(children_df.index))
            else:
                path_weight *= (len(unique_eids)/relation_counts[relation]);
                
            if path_weight <= support:
                continue;
            trie_node_child = TrieNode(label=relation, level=trie_node.level+1, path=trie_node.path+[relation], path_weight=path_weight, filtered_df=children_df)
            task_queue.put(trie_node_child)
            output_queue.put([trie_node.path+[relation], path_weight, rhs_relation_mappings])
        task_queue.task_done()


# In[13]:


# to save the results
def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[14]:


# Python program for insert and search
# operation in a Trie
 
class TrieNode:
     
    # Trie node class
    def __init__(self, label=None, level=0, path=[], path_weight=0.0, filtered_df=None):
        self.children = {}
        self.label = label
        self.level = level
        self.path = path
        self.path_weight = path_weight
        self.filtered_df = filtered_df
        
        


# In[15]:


def main(args):
    init_time_start = time.time()
    # load dataset
    dataset = get_dataset(args.data_path,
                          args.dataset,
                          args.format,
                          args.delimiter,
                          args.data_files,
                          args.has_edge_importance)
    
    # create dgl graph
    g = ConstructGraph(dataset.train, dataset.n_entities, args)
    
    # initialize task_queue, we generate the composition of relations as a parallel BFS search
    task_queue = mp.JoinableQueue()   # this is where we are going to store input data
    output_queue = mp.Queue()   # this is where we are going to store output
    relation_counts = {}
    # lets create a pd dataframe, makes it easy and efficient to create children
    eids = torch.tensor(range(g.number_of_edges()))
    src, dst = g.find_edges(eids)
    graph_df = pd.DataFrame({'src': src.numpy(), 'dst': dst.numpy(), 'tid': g.edata['tid'].numpy(), 'direction':g.edata['direction'].numpy(), 'eid':eids.numpy()})
    
    for index, df in graph_df.rename(columns={"tid": "tid_0", "direction": "direction_0", "eid": "eid_0"}).groupby(['tid_0', 'direction_0']):
        
        # count the number of relations
        relation_counts[index] = len(df.index)
        
        # To avoid traversing both a composition+reversed composition, we only add seeds with single directions
#         if index[1] == -1:
#             continue
        tree_node = TrieNode(label=index, level=1, path=[index], path_weight=1.0, filtered_df=df[['src', 'dst', 'tid_0', 'direction_0', 'eid_0']])
        task_queue.put(tree_node)  # Send a lot of stuff for workers tasks
        output_queue.put([[index], 1.0, {(index):1.0}])
    
    print( "Initial steps took %s seconds" % (time.time() - init_time_start))

    # create num_worker processes, which takes queues as arguments
    # the data will get in through the queues
    # daemonize it
    processes = []
    for i in range(args.num_workers):
        if args.join_type == 'pre':
            worker_process = mp.Process(target=populate_worker_pre_join, args=(task_queue,output_queue,relation_counts,args.max_level,args.support,args.confidence,args.sampling_prob,args.add_correction,graph_df), daemon=True, name='worker_process_{}'.format(i))
        elif args.join_type == 'post':
            worker_process = mp.Process(target=populate_worker_post_join, args=(task_queue,output_queue,relation_counts,args.max_level,args.support,args.confidence,args.sampling_prob,args.add_correction,graph_df), daemon=True, name='worker_process_{}'.format(i))
        else:
            continue;
        worker_process.start()        # Launch reader() as a separate python process
        processes.append(worker_process)

    print([x.name for x in processes])

    _start = time.time()
    # wait till everything is processed
    while task_queue.qsize() > 0:
        print('Queue has {} unexplored nodes'.format(task_queue.qsize()))
        time.sleep(args.log_interval)
    task_queue.join()
    print( "Took %s seconds" % (time.time() - _start))
    
    
    write_start = time.time()
    relation_to_weight = {}
    while output_queue.qsize() > 0:
        msg = output_queue.get()         # Read from the queue
        relation_to_weight[tuple(msg[0])] = (msg[1], msg[2])
    save_obj(relation_to_weight, args.save_path)
    print( "Took %s seconds to write the composition relatons" % (time.time() - write_start))


# In[ ]:


if __name__ == '__main__':
    mp.set_start_method('fork', True)
    args = ArgParser().parse_args()
    main(args)