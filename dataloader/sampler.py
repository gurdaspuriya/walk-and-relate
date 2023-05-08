# -*- coding: utf-8 -*-
#
# sampler.py
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
import copy
import math
import random
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import sys
import pickle
import time
import torch
import pandas as pd # For sampling edge in case two nodes have more than one edge in between
from torch.utils.data import DataLoader, WeightedRandomSampler

from dgl.base import NID, EID
import time


from collections import Counter

def RandomPartition(edges, n, has_importance=False):
    """This partitions a list of edges randomly across n partitions

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    """
    if has_importance:
        heads, rels, tails, e_impts = edges
    else:
        heads, rels, tails = edges
    print('random partition {} edges into {} parts'.format(len(heads), n))
    idx = np.random.permutation(len(heads))
    heads[:] = heads[idx]
    rels[:] = rels[idx]
    tails[:] = tails[idx]
    if has_importance:
        e_impts[:] = e_impts[idx]

    part_size = int(math.ceil(len(idx) / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), len(idx))
        parts.append(idx[start:end])
        print('part {} has {} edges'.format(i, len(parts[-1])))
    return parts

# def ConstructGraph(edges, n_entities, args):
#     """Construct Graph for training

#     Parameters
#     ----------
#     edges : (heads, rels, tails) triple
#         Edge list
#     n_entities : int
#         number of entities
#     args :
#         Global configs.
#     """
#     if args.has_edge_importance:
#         src, etype_id, dst, e_impts = edges
#     else:
#         src, etype_id, dst = edges
#     coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
#     g = dgl.DGLGraph(coo, readonly=False, multigraph=True, sort_csr=True)
#     g.add_edges(g.edges()[1], g.edges()[0])
#     etype_direction = torch.cat((F.tensor([1]*len(etype_id), F.int64), F.tensor([-1]*len(etype_id), F.int64)) ,dim=0)
#     etype_id = torch.cat((F.tensor(etype_id, F.int64), F.tensor(etype_id, F.int64)) ,dim=0)
#     g.edata['tid'] = etype_id
#     g.edata['direction'] = etype_direction
#     if args.has_edge_importance:
#         e_impts = torch.cat((F.tensor(e_impts, F.float), F.tensor(e_impts, F.float)) ,dim=0)
#         g.edata['impts'] = e_impts
#     g.readonly()
    
    
# #     g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
# #     g.edata['tid'] = F.tensor(etype_id, F.int64)
# #     if args.has_edge_importance:
# #         g.edata['impts'] = F.tensor(e_impts, F.float32)
#     return g


def ConstructGraph(edges, n_entities, args):
    """Construct Graph for training

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list
    n_entities : int
        number of entities
    args :
        Global configs.
    """
    if args.has_edge_importance:
        src, etype_id, dst, e_impts = edges
    else:
        src, etype_id, dst = edges
    
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
    
    # stalegraph to use edgesampler, which only works on homogeneous graphs and not heterographs
    g = dgl.DGLGraphStale(coo)
    g.edata['tid'] = F.tensor(etype_id, F.int64)
    if args.has_edge_importance:
        g.edata['impts'] = F.tensor(e_impts, F.float32)
    g.readonly()
    return g


class TrainDataset(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """
    def __init__(self, dataset, args, ranks=64, has_importance=False):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)
        if ranks > 1:
            self.edge_parts = RandomPartition(triples, ranks, has_importance=has_importance)
            self.cross_part = True
        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = ConstructGraph(triples, dataset.n_entities, args)

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        assert batch_size % neg_sample_size == 0, 'batch_size should be divisible by B'
        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank]),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)
    
def find_connected_nodes(g):
    nodes = g.out_degrees().nonzero().squeeze(-1)
    return nodes

def shuffle_walks(walks):
    seeds = torch.randperm(walks.size()[0])
    return walks[seeds]

class RandompWalkGenarator(object):
    def __init__(self, g, seeds, walk_length, window_size, batch_size, has_edge_importance):
        """ random walk sampler 
        
        Parameter
        ---------
        g dgl.Graph : the input graph
        seeds torch.LongTensor : starting nodes
        walk_length int : walk length
        window_size int : window size
        batch_size int : batch size
        """
        # for randomwlak sampler, we need graph, not stalegraph
        self.g = dgl.graph(g.edges(order='eid'))
        etype_id = g.edata['tid']
        etype_direction = torch.cat((F.tensor([1]*len(etype_id), F.int64), F.tensor([-1]*len(etype_id), F.int64)) ,dim=0)
        etype_id = torch.cat((etype_id, etype_id) ,dim=0)
        if 'impts' in self.g.edata:
            e_impts = self.g.edata['impts']
            e_impts = torch.cat((e_impts, e_impts) ,dim=0)
            
        self.g.add_edges(self.g.edges()[1], self.g.edges()[0])
        self.g.edata['tid'] = etype_id
        self.g.edata['direction'] = etype_direction
        if 'impts' in g.edata:
            self.g.edata['impts'] = e_impts
            
            
#         self.g.readonly()
#         etype_direction = F.tensor([1]*len(etype_id), F.int64)
#         self.g.edata['direction'] = etype_direction
#         self.g.edata['tid'] = etype_id
    
        
        
        self.seeds = seeds
        self.walk_length = walk_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.has_edge_importance = has_edge_importance
        # indexes to select positive/negative node pairs from batch_walks
        self.index_posu, self.index_posv, self.index_u_edge, self.index_v_edge, self.index_edge_start, self.index_edge_end, self.index_sampled_eids, self.num_edges_path = self.init_pos_index(
            self.walk_length,
            self.window_size,
            self.batch_size)
        
    def init_pos_index(self, walk_length, window_size, batch_size):
        ''' select embedding of positive nodes from a batch of node embeddings

        Return
        ------
        index_emb_posu torch.LongTensor : the indices of u_embeddings
        index_emb_posv torch.LongTensor : the indices of v_embeddings

        Usage
        -----
        # emb_u.shape: [batch_size * walk_length, dim]
        batch_emb2posu = torch.index_select(emb_u, 0, index_emb_posu)
        '''
        idx_list_u = []
        idx_list_v = []
        idx_list_e_start = []
        idx_list_e_end = []
        idx_list_u_edge = []
        idx_list_v_edge = []

        for b in range(batch_size):
            for i in range(walk_length-1):
                idx_list_u_edge.append(i + b * walk_length)
                idx_list_v_edge.append(i + b * walk_length + 1)

        for b in range(batch_size):
            for i in range(walk_length):
                for j in range(window_size):
                    if (i-window_size + j) >= 0:
                        idx_list_u.append(i-window_size + j + b * walk_length)
                        idx_list_v.append(i + b * walk_length)
                        idx_list_e_end.append(i + b * (walk_length - 1) - 1)
                        idx_list_e_start.append(i-window_size + j + b * (walk_length - 1))

                for j in range(window_size):
                    if (i + 1 + j) < walk_length:
                        idx_list_v.append(i + 1 + j + b * walk_length)
                        idx_list_u.append(i + b * walk_length)
                        idx_list_e_start.append(i + b * (walk_length - 1))
                        idx_list_e_end.append(j + i + b * (walk_length - 1))

                        
        index_posu = torch.LongTensor(idx_list_u)
        index_posv = torch.LongTensor(idx_list_v)
        index_u_edge = torch.LongTensor(idx_list_u_edge)
        index_v_edge = torch.LongTensor(idx_list_v_edge)
        index_edge_start = torch.LongTensor(idx_list_e_start)
        index_edge_end = torch.LongTensor(idx_list_e_end)
        
        index_sampled_eids = torch.zeros((len(index_posu), self.window_size), dtype=torch.long)
        for i in range(len(index_posu)):
            index_sampled_eids[i, :index_edge_end[i]-index_edge_start[i]+1] = torch.LongTensor(list(range(index_edge_start[i], index_edge_end[i]+1, 1)))
        index_sampled_eids = index_sampled_eids.view(-1)        
        num_edges_path = index_edge_end-index_edge_start+1

        return index_posu, index_posv, index_u_edge, index_v_edge, index_edge_start, index_edge_end, index_sampled_eids, num_edges_path
    
    def sample(self, seeds):
        seeds = torch.stack(seeds, dim=-1)
        
        if 'impts' in self.g.edata:
            node_id_traces, edge_id_traces, _ = dgl.sampling.random_walk(self.g, seeds, length=self.walk_length-1, prob='impts', return_eids=True)
        else:
            node_id_traces, edge_id_traces, _ = dgl.sampling.random_walk(self.g, seeds, length=self.walk_length-1, return_eids=True)

        
        
        relation_ids = self.g.edata['tid'][edge_id_traces]
        relation_direction = self.g.edata['direction'][edge_id_traces]
        
        
        return node_id_traces, edge_id_traces, relation_ids, relation_direction
    
    def construct_graph_from_walks(self, walks):
        bs = len(walks)
        if bs < self.batch_size:
            index_posu, index_posv, index_u_edge, index_v_edge, index_edge_start, index_edge_end, index_sampled_eids, num_edges_path = self.init_pos_index(
                self.walk_length, 
                self.window_size, 
                bs)
        else:
            index_posu = self.index_posu
            index_posv = self.index_posv
            index_u_edge = self.index_u_edge
            index_v_edge = self.index_v_edge
            index_edge_start = self.index_edge_start
            index_edge_end = self.index_edge_end
            index_sampled_eids = self.index_sampled_eids
            num_edges_path = self.num_edges_path
        concatenated_walks = walks.view(-1)
        sorted_indices, inverse_indices = torch.unique(concatenated_walks, sorted=True, return_inverse=True)
        
        src_nodes = torch.index_select(inverse_indices, 0, index_posu)
        dst_nodes = torch.index_select(inverse_indices, 0, index_posv)
        sampled_graph = dgl.DGLGraph((src_nodes, dst_nodes))
        
        original_edge_index_u = torch.index_select(concatenated_walks, 0, index_u_edge)
        original_edge_index_v = torch.index_select(concatenated_walks, 0, index_v_edge)
        
        # In case of multiple edges, we will sample with the edge weights, will request DGL to hashis feature in random walk
        temp_u, temp_v, original_eids = self.g.edge_ids(original_edge_index_u, original_edge_index_v, return_uv=True)
        # Using pandas df for readymade efficient implementaton
        df = pd.DataFrame({'u': temp_u, 'v': temp_v, 'eid':original_eids})
        if self.has_edge_importance:
            weights = self.g.edata['impts'][original_eids].tolist()
        else:
            weights = [1]*len(original_eids)
            
        sampled_df = df.groupby(["u", "v"]).sample(n=1, weights=weights).set_index(['u', 'v'])
        sampled_original_eids = torch.LongTensor(sampled_df.loc[np.array(zip(original_edge_index_u.tolist(), original_edge_index_v.tolist())) , 'eid'].values)
        
        
        original_eids_path = torch.index_select(sampled_original_eids, 0, index_sampled_eids).view(len(index_posu), -1)
        relation_ids_path = torch.index_select(sampled_original_eids, 0, index_sampled_eids).view(len(index_posu), -1)
        relation_direction_path = torch.index_select(sampled_original_eids, 0, index_sampled_eids).view(len(index_posu), -1)
        
        sampled_graph.ndata['id'] = sorted_indices
#         sampled_graph.edata['eid_path'] = original_eids_path
        sampled_graph.edata['path_len'] = num_edges_path
        return sampled_graph, sampled_original_eids
        
    
class TrainDataset_v2(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """
    def __init__(self, dataset, args, ranks=64, has_importance=False):
        triples = dataset.train
        num_train = len(triples[0])
        
        
        unique, counts = np.unique(np.concatenate([dataset.train[0], dataset.train[2]]), return_counts=True)
        min_count = np.min(counts)
        max_count = np.max(counts)
        normalization_denominator = max_count - min_count
        sparsity_factor = np.array([1 - (counts[i] - min_count)/normalization_denominator for i in range(len(counts))])
        selected_nids = [unique[i] for i in range(len(unique) )if sparsity_factor[i] > 0.995]
    
        self.args = args
        self.walk_length = args.walk_length
        self.window_size = args.window_size
        self.batch_size = args.batch_size
        self.has_edge_importance = self.args.has_edge_importance
        print('|Train|:', num_train)
        if ranks > 1:
            self.edge_parts = RandomPartition(triples, ranks, has_importance=has_importance)
            self.cross_part = True
        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = ConstructGraph(triples, dataset.n_entities, args)
        
        self.num_nodes = self.g.number_of_nodes()

        # random walk seeds
        start = time.time()
#         self.valid_seeds = find_connected_nodes(self.g)
#         self.valid_seeds = F.tensor(np.array([_ for _ in range(self.g.number_of_nodes())]), F.int64)
        self.valid_seeds = F.tensor(np.array(selected_nids), F.int64)
        if len(self.valid_seeds) != self.num_nodes:
            print("WARNING: The node ids are not serial. Some nodes are invalid.")
        
        seeds = F.tensor(self.valid_seeds, F.int64)
        self.seeds = F.split(shuffle_walks(self.valid_seeds), 
            int(np.ceil(len(self.valid_seeds) / ranks)), 
            0)
        end = time.time()
        t = end - start
        print("%d seeds in %.2fs" % (len(seeds), t))
        
        # number of positive node pairs in a sequence
        self.num_pos = int(2 * self.args.walk_length * self.args.window_size\
            - self.args.window_size * (self.args.window_size + 1))//2
        
        print(self.num_pos)

    def create_random_walk_sampler(self, batch_size, num_workers=1, shuffle=True, rank=0):
        """Create random walk sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
            Random walk sampler
        """
        
        RandomWalkSampler = RandompWalkGenarator(self.g, seeds=self.seeds[rank], walk_length=self.walk_length, window_size=self.window_size, batch_size=self.batch_size, has_edge_importance=self.has_edge_importance)
        
        extended_seeds = []
        for a_, size in zip(RandomWalkSampler.seeds, self.g.out_degrees(RandomWalkSampler.seeds)+self.g.in_degrees(RandomWalkSampler.seeds)):
            extended_seeds.append(a_.repeat(size))
        extended_seeds = torch.cat(extended_seeds)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        dataloader = DataLoader(
            dataset=extended_seeds,
            batch_size=batch_size,
            collate_fn=RandomWalkSampler.sample,
            shuffle=shuffle,
            drop_last=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g
            )

#         weights = self.g.out_degrees(RandomWalkSampler.seeds)
#         weights[weights > 1] = 1
        
#         dataloader = DataLoader(
#             dataset=RandomWalkSampler.seeds,
#             batch_size=batch_size,
#             collate_fn=RandomWalkSampler.sample,
# #             sampler = WeightedRandomSampler(weights=self.g.out_degrees(RandomWalkSampler.seeds)+self.g.in_degrees(RandomWalkSampler.seeds), num_samples=len(RandomWalkSampler.seeds)),
#             sampler = WeightedRandomSampler(weights=self.g.out_degrees(RandomWalkSampler.seeds), num_samples=len(RandomWalkSampler.seeds), replacement=True),
# #             sampler = WeightedRandomSampler(weights=weights, num_samples=len(RandomWalkSampler.seeds)),
#             drop_last=False,
#             num_workers=num_workers,
#             worker_init_fn=seed_worker,
#             generator=g
#             )
        print('yo', rank)
        return dataloader

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=1,
                       shuffle=True, exclude_positive=False, rank=0):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        assert (batch_size*self.num_pos) % neg_sample_size == 0, 'batch_size should be divisible by B'
        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank]),
                           batch_size=batch_size*self.num_pos,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)

class ChunkNegEdgeSubgraph(dgl.DGLGraphStale):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid

def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)

class EvalSampler(object):
    """Sampler for validation and testing

    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    filter_false_neg : bool
        If True, exlucde true positive edges in sampled negative edges
        If False, return all sampled negative edges even there are positive edges
        Default: True
    """
    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg)


        self.sampler_iter = iter(self.sampler)
#         self.sampler_iter = self.iterator_temp(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size

    def __iter__(self):
        return self

#     def __next__(self):
#         """Get next batch

#         Returns
#         -------
#         DGLGraph
#             Sampled positive graph
#         ChunkNegEdgeSubgraph
#             Negative graph wrapper
#         """
#         return next(self.sampler_iter)



    def __next__(self):
        """Get next batch

        Returns
        -------
        DGLGraph
            Sampled positive graph
        ChunkNegEdgeSubgraph
            Negative graph wrapper
        """
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g,
                                        self.neg_chunk_size,
                                        self.neg_sample_size,
                                        'chunk' in self.mode,
                                        self.neg_head,
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

# #         while True:
#         for pos_g, neg_g in self.sampler:
#             if self.filter_false_neg:
#                 neg_positive = neg_g.edata['false_neg']
#             neg_g = create_neg_subgraph(pos_g, neg_g,
#                                         self.neg_chunk_size,
#                                         self.neg_sample_size,
#                                         'chunk' in self.mode,
#                                         self.neg_head,
#                                         self.g.number_of_nodes())
#             if neg_g is None:
#                 continue

#             pos_g.ndata['id'] = pos_g.parent_nid
#             neg_g.ndata['id'] = neg_g.parent_nid
#             pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
#             if self.filter_false_neg:
#                 neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
#             yield pos_g, neg_g


    def reset(self):
        """Reset the sampler
        """
        self.sampler_iter = iter(self.sampler)
#         self.sampler_iter = self.iterator_temp(self.sampler)
        return self

class WikiEvalSampler(object):
    """Sampler for validation and testing for wikikg90M dataset

    Parameters
    ----------
    edges : tensor
        sampled test data
    batch_size : int
        Batch size of each mini batch.
    mode : str
        Sampling mode.
    """
    def __init__(self, edges, batch_size, mode):
        self.edges = edges
        self.batch_size = batch_size
        self.mode = mode
        self.neg_head = 'head' in mode
        self.cnt = 0
        self.mode = 'h,r->t'
        self.num_edges = len(self.edges['h,r->t']['hr'])

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        tensor of size (batch_size, 2)
            sampled head and relation pair
        tensor of size (batchsize, 1)
            the index of the true tail entity
        tensor of size (bath_size, 1001)
            candidates for the tail entities (1001 candidates in total, out of which one is a positive entity)
        """
        if self.cnt == self.num_edges:
            raise StopIteration
        beg = self.cnt
        if self.cnt + self.batch_size > self.num_edges:
            self.cnt = self.num_edges
        else:
            self.cnt += self.batch_size
        return F.tensor(self.edges['h,r->t']['hr'][beg:self.cnt], F.int64), F.tensor(self.edges['h,r->t']['t_correct_index'][beg:self.cnt], F.int64), F.tensor(self.edges['h,r->t']['t_candidate'][beg:self.cnt], F.int64)

    def reset(self):
        """Reset the sampler
        """
        self.cnt = 0
        return self


class EvalDataset(object):
    """Dataset for validation or testing

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """
    def __init__(self, dataset, args):
        src = [dataset.train[0]]
        etype_id = [dataset.train[1]]
        dst = [dataset.train[2]]
        self.num_train = len(dataset.train[0])
        if args.dataset == "wikikg90M":
            self.valid_dict = dataset.valid
            self.num_valid = len(self.valid_dict['h,r->t']['hr'])
        elif dataset.valid is not None:
            src.append(dataset.valid[0])
            etype_id.append(dataset.valid[1])
            dst.append(dataset.valid[2])
            self.num_valid = len(dataset.valid[0])
        else:
            self.num_valid = 0
        if args.dataset == "wikikg90M":
            self.test_dict = dataset.test
            self.num_test = len(self.test_dict['h,r->t']['hr'])
        elif dataset.test is not None:
            src.append(dataset.test[0])
            etype_id.append(dataset.test[1])
            dst.append(dataset.test[2])
            self.num_test = len(dataset.test[0])
        else:
            self.num_test = 0

        if args.dataset == "wikikg90M":
            print('|valid|:', self.num_valid)
            print('|test|:', self.num_test)
            return

        assert len(src) > 1, "we need to have at least validation set or test set."

        src = np.concatenate(src)
        etype_id = np.concatenate(etype_id)
        dst = np.concatenate(dst)

        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                    shape=[dataset.n_entities, dataset.n_entities])
        # stalegraph as we want a homegoneous graph to use the dgl edgesampler
        g = dgl.DGLGraphStale(coo)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        g.readonly()
        self.g = g

        if args.eval_percent < 1:
            self.valid = np.random.randint(0, self.num_valid,
                    size=(int(self.num_valid * args.eval_percent),)) + self.num_train
        else:
            self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

        if args.eval_percent < 1:
            self.test = np.random.randint(0, self.num_test,
                    size=(int(self.num_test * args.eval_percent,)))
            self.test += self.num_train + self.num_valid
        else:
            self.test = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        print('|test|:', len(self.test))

    def get_edges(self, eval_type):
        """ Get all edges in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def get_dicts(self, eval_type):
        """ Get all edges dict in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        dict
            all edges
        """
        if eval_type == 'valid':
            return self.valid_dict
        elif eval_type == 'test':
            return self.test_dict
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=0, rank=0, ranks=1):
        """Create sampler for validation or testing

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)

    def create_sampler_wikikg90M(self, eval_type, batch_size, mode='head', rank=0, ranks=1):
        """Create sampler for validation and testing of wikikg90M dataset.

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        mode : str
            Sampling mode.
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_dicts(eval_type)
        new_edges = {}

        assert 'tail' in mode

        """
        This function will split the edges into total number of partitions parts. And then calculate the 
        corresponding begin and end index for each part to create evaluate sampler.
        """
        beg = edges['h,r->t']['hr'].shape[0] * rank // ranks
        end = min(edges['h,r->t']['hr'].shape[0] * (rank + 1) // ranks, edges['h,r->t']['hr'].shape[0])
        new_edges['h,r->t'] = {'hr': edges['h,r->t']['hr'][beg:end],
                                't_candidate': edges['h,r->t']['t_candidate'][beg:end],
                            #    't_correct_index': edges['h,r->t']['t_correct_index'][beg:end]
                                }
        if 't_correct_index' in edges['h,r->t']:
            new_edges['h,r->t']['t_correct_index'] = edges['h,r->t']['t_correct_index'][beg:end]
        else:
            new_edges['h,r->t']['t_correct_index'] = np.zeros(end-beg, dtype=np.short)

        return WikiEvalSampler(new_edges, batch_size, mode)


class NewBidirectionalOneShotIterator:
    """Grouped sampler iterator

    Parameters
    ----------
    dataloader_walk : random walk sampler
    dataloader_head : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in head mode
    dataloader_tail : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in tail mode
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    neg_sample_size : int
        How many negative edges sampled for each node.
    is_chunked : bool
        If True, the sampled batch is chunked.
    num_nodes : int
        Total number of nodes in the whole graph.
    """
    def __init__(self, dataloader_walk, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes, walk_length, window_size, batch_size, relation_confidence,
                    rule_mapping=None,
                    composition_to_id_mapping=None, has_edge_importance=False, g=None):
        
        self.walk_length = walk_length
        self.window_size = window_size
        self.batch_size = batch_size
        self.sampler_walk = dataloader_walk
        self.iterator_walk = self.random_walk_iterator(self.sampler_walk)
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        
        # Mapping from (composed) relations to their confidence
        self.relation_confidence = relation_confidence
        
        # Mapping from (composed) relations to existing relations (rules)
        self.rule_mapping = rule_mapping
        
        # Mapping from (composed) relations to new relations
        self.composition_to_id_mapping = composition_to_id_mapping
        
        self.iterator_head = self.one_shot_iterator(self.iterator_walk, dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes, has_edge_importance)
        self.iterator_tail = self.one_shot_iterator(self.iterator_walk, dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes, has_edge_importance)
        self.step = 0
        
        self.index_posu, self.index_posv, self.index_edge_start, self.index_edge_end, self.index_sampled_eids, self.reversed_index_sampled_eids, self.num_edges_path = self.init_pos_index(
            self.walk_length,
            self.window_size,
            self.batch_size)
        
        
        # number of positive node pairs in a sequence
        self.num_pos = int(2 * self.walk_length * self.window_size\
            - self.window_size * (self.window_size + 1))//2

        # for randomwalk sampler, we need graph, not stalegraph
        
        edge_list = g.edges(order='eid')
        self.g = dgl.graph(edge_list)
        etype_id = g.edata['tid']
        etype_direction = torch.cat((F.tensor([1]*len(etype_id), F.int64), F.tensor([-1]*len(etype_id), F.int64)) ,dim=0)
        etype_id = torch.cat((etype_id, etype_id) ,dim=0)
        if 'impts' in g.edata:
            e_impts = g.edata['impts']
            e_impts = torch.cat((e_impts, e_impts) ,dim=0)
            
            
        self.g.add_edges(edge_list[1], edge_list[0])
        self.g.edata['tid'] = etype_id
        self.g.edata['direction'] = etype_direction
        if 'impts' in g.edata:
            self.g.edata['impts'] = e_impts
            
        unique, counts = np.unique(np.concatenate([edge_list[0].numpy(), edge_list[1].numpy()]), return_counts=True)
        min_count = np.min(counts)
        max_count = np.max(counts)
        normalization_denominator = max_count - min_count
        sparsity_factor = np.array([1 - (counts[i] - min_count)/normalization_denominator for i in range(len(counts))])
        self.selected_nids = [unique[i] for i in range(len(unique) )if sparsity_factor[i] > 0.995]
        self.selected_nids_tensor = torch.LongTensor([unique[i] for i in range(len(unique) )if sparsity_factor[i] > 0.995])
        self.pos_counter = Counter()
        self.walk_counter = Counter()
        self.total_counter = Counter(self.g.edata['tid'].tolist())
         

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            walk_g, pos_g, neg_g = next(self.iterator_head)
        else:
            walk_g, pos_g, neg_g = next(self.iterator_tail)
        return walk_g, pos_g, neg_g
    
        
    
    def init_pos_index(self, walk_length, window_size, batch_size):
        ''' select indices to ease the creation of positive graph based on random walks

        Return
        ------
        index_emb_posu torch.LongTensor : the indices of src nodes of the new graph based on random walks
        index_emb_posv torch.LongTensor : the indices of dst nodes of the new graph based on random walks
        index_edge_start torch.LongTensor : the indices of src edge of the random path sequence
        index_edge_end torch.LongTensor : the indices of dst edge of the random path sequence
        index_sampled_eids torch.LongTensor : the indices of edges in the random walk sequence
        reversed_index_sampled_eids torch.LongTensor : the indices of edges in the reversed random walk sequence
        num_edges_path torch.LongTensor : the lengths the random walk sequences
        '''
        idx_list_u = []
        idx_list_v = []
        idx_list_e_start = []
        idx_list_e_end = []

        for b in range(batch_size):
            for i in range(walk_length):
                for j in range(window_size):
                    if (i-window_size + j) >= 0:
                        idx_list_u.append(i-window_size + j + b * walk_length)
                        idx_list_v.append(i + b * walk_length)
                        idx_list_e_end.append(i + b * (walk_length - 1) - 1)
                        idx_list_e_start.append(i-window_size + j + b * (walk_length - 1))

#                 for j in range(window_size):
#                     if (i + 1 + j) < walk_length:
#                         idx_list_v.append(i + 1 + j + b * walk_length)
#                         idx_list_u.append(i + b * walk_length)
#                         idx_list_e_start.append(i + b * (walk_length - 1))
#                         idx_list_e_end.append(j + i + b * (walk_length - 1))

                        
        index_posu = torch.LongTensor(idx_list_u)
        index_posv = torch.LongTensor(idx_list_v)
        index_edge_start = torch.LongTensor(idx_list_e_start)
        index_edge_end = torch.LongTensor(idx_list_e_end)
        
        index_sampled_eids = torch.zeros((len(index_posu), self.window_size-1), dtype=torch.long)
        index_sampled_eids = torch.randint(low=0, high=batch_size*(walk_length-1), size=(len(index_posu), min(self.walk_length-1, self.window_size)), dtype=torch.long)
        # Reversed index_sampled_eids, useful in disambiguating if composition a is simply reverse of composition b
        reversed_index_sampled_eids = torch.zeros((len(index_posu), self.window_size-1), dtype=torch.long)
        reversed_index_sampled_eids = torch.randint(low=0, high=batch_size*(walk_length-1), size=(len(index_posu), min(self.walk_length-1, self.window_size)), dtype=torch.long)
        for i in range(len(index_posu)):
            index_sampled_eids[i, :index_edge_end[i]-index_edge_start[i]+1] = torch.LongTensor(list(range(index_edge_start[i], index_edge_end[i]+1, 1)))
            reversed_index_sampled_eids[i, :index_edge_end[i]-index_edge_start[i]+1] = torch.LongTensor(list(reversed(range(index_edge_start[i], index_edge_end[i]+1, 1))))
            
        index_sampled_eids = index_sampled_eids.view(-1)        
        reversed_index_sampled_eids = reversed_index_sampled_eids.view(-1)
        
        num_edges_path = index_edge_end-index_edge_start+1


        return index_posu, index_posv, index_edge_start, index_edge_end, index_sampled_eids, reversed_index_sampled_eids, num_edges_path

    def one_shot_iterator(self, dataloader_walk, dataloader_edge, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes, has_edge_importance=False):
        prev_pos_g = None
        while True:
            for pos_g, neg_g in dataloader_edge:
                if prev_pos_g is None:
                    prev_pos_g = pos_g
                    continue;
                prev_pos_g, pos_g = pos_g, prev_pos_g
                
                # Lets remove the last batches, as the negative sample sizes may not be compatible
                if pos_g.number_of_edges() != self.batch_size*self.num_pos:
                    continue
                if neg_g.number_of_edges() != self.batch_size*self.num_pos*int(neg_sample_size/neg_chunk_size):
                    continue
                start1 = time.time()
                
                neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                            is_chunked, neg_head, num_nodes)
                if neg_g is None:
                    continue

                pos_g.ndata['id'] = pos_g.parent_nid
                neg_g.ndata['id'] = neg_g.parent_nid
                pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                if has_edge_importance:
                    pos_g.edata['impts'] = pos_g._parent.edata['impts'][pos_g.parent_eid]
#                 yield pos_g, neg_g
                
                walk_g = next(dataloader_walk)
                
                # Lets remove the last batches, as the negative sample sizes may not be compatible
                if walk_g.number_of_edges() != self.batch_size*self.num_pos:
                    continue
                if walk_g.number_of_edges() != self.batch_size*self.num_pos*int(neg_sample_size/neg_chunk_size):
                    continue
                    
                
                    
#                 pos_g_edges = pos_g.edges(order='eid')
#                 walk_g_edges = walk_g.edges(order='eid')
#                 pos_g_triples = set(list(zip(pos_g_edges[0].tolist(), pos_g_edges[1].tolist())))
#                 walk_g_triples = set(list(zip(walk_g_edges[0].tolist(), walk_g_edges[1].tolist())))
                    
#                 print(pos_g.number_of_edges(), walk_g.number_of_edges(), len(pos_g_triples), len(walk_g_triples), pos_g.number_of_nodes(), walk_g.number_of_nodes(), len(set(pos_g_edges[0].tolist())), len(set(walk_g_edges[0].tolist())), len(set(pos_g_edges[1].tolist())), len(set(walk_g_edges[1].tolist())))
#                 x = self.g.edge_ids(pos_g.ndata['id'][pos_g_edges[0]], pos_g.ndata['id'][pos_g_edges[1]])
#                 y = self.g.edge_ids(walk_g.ndata['id'][walk_g_edges[0]], walk_g.ndata['id'][walk_g_edges[1]])
#                 self.pos_counter.update(Counter(pos_g.edata['id'].tolist()))
#                 self.walk_counter.update(Counter(walk_g.edata['id'].tolist()))
#                 print(self.pos_counter, self.walk_counter, self.total_counter, self.g.number_of_edges())
                
#                 # Lets remove the last batches, as the negative sample sizes may not be compatible
#                 if pos_g.number_of_edges() != self.batch_size*self.num_pos:
#                     continue
#                 if neg_g.number_of_edges() != self.batch_size*self.num_pos*int(neg_sample_size/neg_chunk_size):
#                     continue
#                 neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
#                                             is_chunked, neg_head, num_nodes)
#                 if neg_g is None:
#                     continue

#                 neg_g.ndata['id'] = neg_g.parent_nid
#                 if has_edge_importance:
#                     pos_g.edata['impts'] = pos_g._parent.edata['impts'][pos_g.parent_eid]
                    
#                 print(pos_g.number_of_nodes(), pos_g.number_of_edges(), neg_g.number_of_nodes(), neg_g.number_of_edges(), walk_g.number_of_nodes(), walk_g.number_of_edges(), flush=True)
#                 if pos_g.number_of_edges() != walk_g.number_of_edges():
#                     continue
                
                yield walk_g, pos_g, neg_g

    def random_walk_iterator(self, dataloader):
        while True:
            for walks, eid_traces, relation_traces, direction_traces in dataloader:
                bs = len(walks)
                if bs < self.batch_size:
                    index_posu, index_posv, index_edge_start, index_edge_end, index_sampled_eids, reversed_index_sampled_eids, num_edges_path = self.init_pos_index(
                        self.walk_length, 
                        self.window_size, 
                        bs)
                else:
                    index_posu = self.index_posu
                    index_posv = self.index_posv
                    index_edge_start = self.index_edge_start
                    index_edge_end = self.index_edge_end
                    index_sampled_eids = self.index_sampled_eids
                    reversed_index_sampled_eids = self.reversed_index_sampled_eids
                    # We change this if we use rules
                    num_edges_path = copy.deepcopy(self.num_edges_path)
                    
                concatenated_walks = walks.view(-1)
                
                sorted_indices, inverse_indices = torch.unique(concatenated_walks, sorted=True, return_inverse=True)

                src_nodes = torch.index_select(inverse_indices, 0, index_posu)
                dst_nodes = torch.index_select(inverse_indices, 0, index_posv)

                
#                 eid_traces = eid_traces.view(-1)
                relation_traces = relation_traces.view(-1)
                direction_traces = direction_traces.view(-1)
                

#                 original_eids_path = torch.index_select(eid_traces, 0, index_sampled_eids).view(len(index_posu), -1)
                relation_traces_path = torch.index_select(relation_traces, 0, index_sampled_eids).view(len(index_posu), -1)
                direction_traces_path = torch.index_select(direction_traces, 0, index_sampled_eids).view(len(index_posu), -1)
        
        
                
                # We need to convert to numpy for vectorization/easy indexing, but doing so with tuples wants a hack
                # https://stackoverflow.com/questions/47389447/how-convert-a-list-of-tupes-to-a-numpy-array-of-tuples
                # Though can directly convert to strings too
                relation_direction_tuple_ = np.empty(len(relation_traces), dtype=object)
                relation_direction_tuple_[:] = list(zip(relation_traces.numpy().tolist(), direction_traces.numpy().tolist()))
                relation_direction_tuple = np.empty(len(index_posu), dtype=object)
                relation_direction_tuple[:] = [tuple(_[:num_edges_path[i]]) for i, _ in enumerate(np.take(relation_direction_tuple_, index_sampled_eids.numpy()).reshape(len(index_posu), -1))]
                
                reversed_relation_traces_path = torch.index_select(relation_traces, 0, reversed_index_sampled_eids).view(len(index_posu), -1)
                reversed_direction_traces_path = torch.index_select(-direction_traces, 0, reversed_index_sampled_eids).view(len(index_posu), -1)
                
                reversed_relation_direction_tuple_ = np.empty(len(relation_traces), dtype=object)
                reversed_relation_direction_tuple_[:] = list(zip(relation_traces.numpy().tolist(), (-direction_traces).numpy().tolist()))
                reversed_relation_direction_tuple = np.empty(len(index_posu), dtype=object)
                reversed_relation_direction_tuple[:] = [tuple(_[:num_edges_path[i]]) for i, _ in enumerate(np.take(reversed_relation_direction_tuple_, reversed_index_sampled_eids.numpy()).reshape(len(index_posu), -1))]
                relation_confidence = torch.Tensor([self.relation_confidence[_] for _ in relation_direction_tuple])
                
                rules_to_map = [self.rule_mapping[relation_direction_tuple[i]]  if num_edges_path[i] > 1 else [] for i in range(len(relation_direction_tuple))]
                rule_confidence = [[i[1] for i in j] for j in rules_to_map]
                rule_consequent = [[i[0] for i in j] for j in rules_to_map]
                rule_index = [list(range(len(j))) for j in rules_to_map]
                selected_rule_consequent_index = [random.choices(rule_index[i], weights=rule_confidence[i], k=1)[0] if len(rule_index[i]) > 0 else -1 for i in range(len(rules_to_map)) ]
                selected_rule_consequent = [rule_consequent[i][selected_rule_consequent_index[i]] if len(rule_consequent[i]) > 0 else () for i in range(len(rules_to_map)) ]
#                 selected_rule_consequent = [random.choices(rule_consequent[i], weights=rule_confidence[i], k=1)[0] if len(rule_consequent[i]) > 0 else () for i in range(len(rules_to_map)) ]
                map_rule_flag = [len(rule_consequent[i]) > 0 for i in range(len(rules_to_map))]
                
#                 relation_confidence[map_rule_flag] = 0.0
#                 # We need to recreate these data after mapping the rules
                num_edges_path[map_rule_flag] = 1
                
                
                relation_traces_path[map_rule_flag,0] = F.tensor([_[0] for _ in selected_rule_consequent if len(_) == 2], F.int64)
                direction_traces_path[map_rule_flag,0] = F.tensor([_[1] for _ in selected_rule_consequent if len(_) == 2], F.int64)
                
                reversed_relation_traces_path[map_rule_flag,0] = F.tensor([_[0] for _ in selected_rule_consequent if len(_) == 2], F.int64)
                reversed_direction_traces_path[map_rule_flag,0] = F.tensor([-1*_[1] for _ in selected_rule_consequent if len(_) == 2], F.int64)
                
                relation_confidence[map_rule_flag] = relation_confidence[map_rule_flag]*torch.Tensor([rule_confidence[i][selected_rule_consequent_index[i]] for i in range(len(rules_to_map)) if len(rule_consequent[i]) > 0 ])
                updated_relation_direction_tuple = np.empty(sum(map_rule_flag), dtype=object)
                updated_reversed_relation_direction_tuple = np.empty(sum(map_rule_flag), dtype=object)
                
                updated_relation_direction_tuple[:] = [tuple([_,]) for i, _ in enumerate(selected_rule_consequent) if len(_) == 2]
                relation_direction_tuple[map_rule_flag] = updated_relation_direction_tuple
                
                updated_reversed_relation_direction_tuple[:] = [tuple([(_[0], -1*_[1]),]) for i, _ in enumerate(selected_rule_consequent) if len(_) == 2]
                reversed_relation_direction_tuple[map_rule_flag] = updated_reversed_relation_direction_tuple
                
                change_order_flag = [relation_direction_tuple[i]<reversed_relation_direction_tuple[i] for i in range(len(reversed_relation_direction_tuple))]
    
                # Rearrange the required metadata if the order needs to be changed
                # relation_direction_tuple
                relation_direction_tuple[change_order_flag] = reversed_relation_direction_tuple[change_order_flag]
                # relation traces
                relation_traces_path[change_order_flag,:] = reversed_relation_traces_path[change_order_flag,:]
                # direction traces
                direction_traces_path[change_order_flag,:] = reversed_direction_traces_path[change_order_flag,:]
                # src/dst
                temp = src_nodes[change_order_flag]
                src_nodes[change_order_flag] = dst_nodes[change_order_flag]
                dst_nodes[change_order_flag] = temp
#                 print(num_edges_path[map_rule_flag])
#                 num_edges_path.fill_(1) 
                
                sampled_graph = dgl.DGLGraphStale((src_nodes, dst_nodes))
            
                sampled_graph.ndata['id'] = sorted_indices
                sampled_graph.edata['relation_path'] = relation_traces_path
                sampled_graph.edata['direction_path'] = direction_traces_path
                sampled_graph.edata['path_len'] = num_edges_path
                src_in_id = torch.eq(src_nodes.unsqueeze(1), self.selected_nids_tensor.unsqueeze(0)).any(dim=1)
                dst_in_id = torch.eq(dst_nodes.unsqueeze(1), self.selected_nids_tensor.unsqueeze(0)).any(dim=1)
                len_one_paths = num_edges_path == 1
                
#                 relation_confidence[len_one_paths] = torch.FloatTensor(np.random.randint(2, size=sum(len_one_paths).item()))
#                 relation_confidence[~(src_in_id + dst_in_id + len_one_paths)] = 0.0
#                 relation_confidence[(src_in_id + dst_in_id)] = relation_confidence[(src_in_id + dst_in_id)]*2
#                 print( ~(src_in_id + dst_in_id + len_one_paths), src_in_id, dst_in_id, len_one_paths)
#                 relation_confidence = torch.FloatTensor([relation_confidence[i] if (num_edges_path[i] == 1 or src_nodes[i].item() in self.selected_nids or dst_nodes[i].item() in self.selected_nids) else 0.0 for i in range(len(num_edges_path))])
#                 for i in range(len(relation_confidence)//3):
#                     relation_confidence[3*i+2] = 0.0
                sampled_graph.edata['impts'] = relation_confidence
    
#                 print(sum(relation_confidence))
    
                # For negative loss, it makes sense to use the relations (+head/tail) actually present in the graph
                # So we store another id for an actual relation (not always true when direction is -1)
                sampled_graph.edata['id'] = relation_traces_path[:,0]
                if self.composition_to_id_mapping is not None:
                    sampled_graph.edata['id'] = F.tensor([self.composition_to_id_mapping.get(_, [((0, 1), 1.0)])[0][0][0] for _ in relation_direction_tuple], F.int64)
               
                yield sampled_graph
             