# -*- coding: utf-8 -*-
#
# general_models.py
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
Graph Embedding Model
1. TransE
2. TransR
3. RESCAL
4. DistMult
5. ComplEx
6. RotatE
7. SimplE
"""
import os
import numpy as np
import math
import dgl.backend as F

backend = os.environ.get('DGLBACKEND', 'pytorch')

from .pytorch.tensor_models import logsigmoid
from .pytorch.tensor_models import abs
from .pytorch.tensor_models import masked_select
from .pytorch.tensor_models import get_device, get_dev
from .pytorch.tensor_models import norm
from .pytorch.tensor_models import get_scalar
from .pytorch.tensor_models import reshape
from .pytorch.tensor_models import cuda
from .pytorch.tensor_models import ExternalEmbedding
from .pytorch.tensor_models import InferEmbedding
from .pytorch.score_fun import *
from .pytorch.loss import LossGenerator
from .pytorch.path_embedding import *
DEFAULT_INFER_BATCHSIZE = 2048

EMB_INIT_EPS = 2.0

class InferModel(object):
    def __init__(self, device, model_name, hidden_dim,
        double_entity_emb=False, double_relation_emb=False,
        gamma=0., batch_size=DEFAULT_INFER_BATCHSIZE):
        super(InferModel, self).__init__()

        self.device = device
        self.model_name = model_name
        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        self.entity_emb = InferEmbedding(device)
        self.relation_emb = InferEmbedding(device)
        self.batch_size = batch_size

        if model_name == 'TransE' or model_name == 'TransE_l2':
            self.score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            self.score_func = TransEScore(gamma, 'l1')
        elif model_name == 'TransR':
            assert False, 'Do not support inference of TransR model now.'
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore(relation_dim, entity_dim)
        elif model_name == 'RotatE':
            emb_init = (gamma + EMB_INIT_EPS) / hidden_dim
            self.score_func = RotatEScore(gamma, emb_init)
        elif model_name == 'SimplE':
            self.score_func = SimplEScore()

    def load_emb(self, path, dataset):
        """Load the model.

        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset+'_'+self.model_name)

    def score(self, head, rel, tail, triplet_wise=False):
        head_emb = self.entity_emb(head)
        rel_emb = self.relation_emb(rel)
        tail_emb = self.entity_emb(tail)

        num_head = F.shape(head)[0]
        num_rel = F.shape(rel)[0]
        num_tail = F.shape(tail)[0]

        batch_size = self.batch_size
        score = []
        if triplet_wise:
            class FakeEdge(object):
                def __init__(self, head_emb, rel_emb, tail_emb):
                    self._hobj = {}
                    self._robj = {}
                    self._tobj = {}
                    self._hobj['emb'] = head_emb
                    self._robj['emb'] = rel_emb
                    self._tobj['emb'] = tail_emb

                @property
                def src(self):
                    return self._hobj

                @property
                def dst(self):
                    return self._tobj

                @property
                def data(self):
                    return self._robj

            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                sr_emb = rel_emb[i * batch_size : (i + 1) * batch_size \
                                                  if (i + 1) * batch_size < num_head \
                                                  else num_head]
                st_emb = tail_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                edata = FakeEdge(sh_emb, sr_emb, st_emb)
                score.append(F.copy_to(self.score_func.edge_func(edata)['score'], F.cpu()))
            score = F.cat(score, dim=0)
            return score
        else:
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size : (i + 1) * batch_size \
                                                   if (i + 1) * batch_size < num_head \
                                                   else num_head]
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size : (j + 1) * batch_size \
                                                       if (j + 1) * batch_size < num_tail \
                                                       else num_tail]

                    s_score.append(F.copy_to(self.score_func.infer(sh_emb, rel_emb, st_emb), F.cpu()))
                score.append(F.cat(s_score, dim=2))
            score = F.cat(score, dim=0)
            return F.reshape(score, (num_head * num_rel * num_tail,))

    @property
    def num_entity(self):
        return self.entity_emb.emb.shape[0]

    @property
    def num_rel(self):
        return self.relation_emb.emb.shape[0]

class KEModel(object):
    """ DGL Knowledge Embedding Model.

    Parameters
    ----------
    args:
        Global configs.
    model_name : str
        Which KG model to use, including 'TransE_l1', 'TransE_l2', 'TransR',
        'RESCAL', 'DistMult', 'ComplEx', 'RotatE', 'SimplE'
    n_entities : int
        Num of entities.
    n_relations : int
        Num of relations.
    hidden_dim : int
        Dimension size of embedding.
    gamma : float
        Gamma for score function.
    double_entity_emb : bool
        If True, entity embedding size will be 2 * hidden_dim.
        Default: False
    double_relation_emb : bool
        If True, relation embedding size will be 2 * hidden_dim.
        Default: False
    """
    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False):
        super(KEModel, self).__init__()
        self.args = args
        self.has_edge_importance = args.has_edge_importance
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / hidden_dim
        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        device = get_device(args)

        self.loss_gen = LossGenerator(args, args.loss_genre, args.neg_adversarial_sampling,
                                      args.adversarial_temperature, args.pairwise)

        self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim,
                                            F.cpu() if args.mix_cpu_gpu else device)
        # For RESCAL, relation_emb = relation_dim * entity_dim
        if model_name == 'RESCAL':
            rel_dim = relation_dim * entity_dim
        else:
            rel_dim = relation_dim

        self.rel_dim = rel_dim
        self.entity_dim = entity_dim
        
        self.relation_emb = ExternalEmbedding(args, n_relations, rel_dim,
                                              F.cpu() if args.mix_cpu_gpu else device, num_rels=args.num_bases)
        self.composition_model = None
        
        if args.weight_sharing == 'rnn':
            self.composition_model = PathEmbedding(args, rel_dim, F.cpu() if args.mix_cpu_gpu else device)

        if model_name == 'TransE' or model_name == 'TransE_l2':
            self.score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            self.score_func = TransEScore(gamma, 'l1')
        elif model_name == 'TransR':
            projection_emb = ExternalEmbedding(args,
                                               n_relations,
                                               entity_dim * relation_dim,
                                               F.cpu() if args.mix_cpu_gpu else device, num_rels=args.num_bases)

            self.score_func = TransRScore(gamma, projection_emb, relation_dim, entity_dim)
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore(relation_dim, entity_dim)
        elif model_name == 'RotatE':
            self.score_func = RotatEScore(gamma, self.emb_init)
        elif model_name == 'SimplE':
            self.score_func = SimplEScore()

        self.model_name = model_name
        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)
        self.head_neg_prepare = self.score_func.create_neg_prepare(True)
        self.tail_neg_prepare = self.score_func.create_neg_prepare(False)

        self.reset_parameters()
        

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process embeddings access.
        """
        self.entity_emb.share_memory()
        self.relation_emb.share_memory()

        if self.model_name == 'TransR':
            self.score_func.share_memory()
            
        
        if self.args.weight_sharing == 'rnn':
            self.composition_model.share_memory()

    def save_emb(self, path, dataset):
        """Save the model.

        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.save(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.save(path, dataset+'_'+self.model_name+'_relation')   

        self.score_func.save(path, dataset+'_'+self.model_name)

    def load_emb(self, path, dataset):
        """Load the model.

        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.load(path, dataset+'_'+self.model_name+'_entity')
        self.relation_emb.load(path, dataset+'_'+self.model_name+'_relation')
        self.score_func.load(path, dataset+'_'+self.model_name)

    def reset_parameters(self):
        """Re-initialize the model.
        """
        self.entity_emb.init(self.emb_init)
        self.score_func.reset_parameters()
        self.relation_emb.init(self.emb_init)
        if self.args.weight_sharing == 'rnn':
            self.composition_model.init(self.emb_init)
#         print(th.rand(10, 10))

    def predict_score(self, g):
        """Predict the positive score.

        Parameters
        ----------
        g : DGLGraph
            Graph holding positive edges.

        Returns
        -------
        tensor
            The positive score
        """
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g, to_device=None, gpu_id=-1, trace=False,
                          neg_deg_sample=False):
        """Calculate the negative score.

        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        to_device : func
            Function to move data into device.
        gpu_id : int
            Which gpu to move data to.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: False
        neg_deg_sample : bool
            If True, we use the head and tail nodes of the positive edges to
            construct negative edges.
            Default: False

        Returns
        -------
        tensor
            The negative score
        """
        num_chunks = neg_g.num_chunks
        chunk_size = neg_g.chunk_size
        neg_sample_size = neg_g.neg_sample_size
        mask = F.ones((num_chunks, chunk_size * (neg_sample_size + chunk_size)),
                      dtype=F.float32, ctx=F.context(pos_g.ndata['emb']))
        if neg_g.neg_head:
            neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
            neg_head = self.entity_emb(neg_head_ids, gpu_id, trace)
            head_ids, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                tail_ids = to_device(tail_ids, gpu_id)
            tail = pos_g.ndata['emb'][tail_ids]
            rel = pos_g.edata['emb']

            # When we train a batch, we could use the head nodes of the positive edges to
            # construct negative edges. We construct a negative edge between a positive head
            # node and every positive tail node.
            # When we construct negative edges like this, we know there is one positive
            # edge for a positive head node among the negative edges. We need to mask
            # them.
            if neg_deg_sample:
                head = pos_g.ndata['emb'][head_ids]
                head = head.reshape(num_chunks, chunk_size, -1)
                neg_head = neg_head.reshape(num_chunks, neg_sample_size, -1)
                neg_head = F.cat([head, neg_head], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:,0::(neg_sample_size + 1)] = 0
            neg_head = neg_head.reshape(num_chunks * neg_sample_size, -1)
            neg_head, tail = self.head_neg_prepare(pos_g.edata['id'], num_chunks, neg_head, tail, gpu_id, trace)
            neg_score = self.head_neg_score(neg_head, rel, tail,
                                            num_chunks, chunk_size, neg_sample_size)
        else:
            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            neg_tail = self.entity_emb(neg_tail_ids, gpu_id, trace)
            head_ids, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                head_ids = to_device(head_ids, gpu_id)
            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']

            # This is negative edge construction similar to the above.
            if neg_deg_sample:
                tail = pos_g.ndata['emb'][tail_ids]
                tail = tail.reshape(num_chunks, chunk_size, -1)
                neg_tail = neg_tail.reshape(num_chunks, neg_sample_size, -1)
                neg_tail = F.cat([tail, neg_tail], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:,0::(neg_sample_size + 1)] = 0
            neg_tail = neg_tail.reshape(num_chunks * neg_sample_size, -1)
            head, neg_tail = self.tail_neg_prepare(pos_g.edata['id'], num_chunks, head, neg_tail, gpu_id, trace)
            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)

        if neg_deg_sample:
            neg_g.neg_sample_size = neg_sample_size
            mask = mask.reshape(num_chunks, chunk_size, neg_sample_size)
            return neg_score * mask
        else:
            return neg_score

    def forward_test(self, pos_g, neg_g, logs, gpu_id=-1):
        """Do the forward and generate ranking results.

        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        logs : List
            Where to put results in.
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, False)
        
        if self.args.weight_sharing == 'rnn':
            filler_ones = th.ones(pos_g.number_of_edges(), dtype=th.int8)
            if gpu_id >= 0:
                filler_ones = filler_ones.cuda(gpu_id)
            pos_g.edata['emb'] = self.composition_model(self.relation_emb(pos_g.edata['id'], gpu_id, False).unsqueeze(1), filler_ones, filler_ones.unsqueeze(1), gpu_id, False)
        else:
            pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, False)
        self.score_func.prepare(pos_g, gpu_id, False)

        batch_size = pos_g.number_of_edges()
        pos_scores = self.predict_score(pos_g)
        pos_scores = reshape(pos_scores, batch_size, -1)

        neg_scores = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                            gpu_id=gpu_id, trace=False,
                                            neg_deg_sample=self.args.neg_deg_sample_eval)
        neg_scores = reshape(neg_scores, batch_size, -1)
        # We need to filter the positive edges in the negative graph.
        if self.args.eval_filter:
            filter_bias = reshape(neg_g.edata['bias'], batch_size, -1)
            if gpu_id >= 0:
                filter_bias = cuda(filter_bias, gpu_id)
            # find all indices where it is not false negative sample
            mask = filter_bias != -1

        # To compute the rank of a positive edge among all negative edges,
        # we need to know how many negative edges have higher scores than
        # the positive edge.
        for i in range(batch_size):
            if self.args.eval_filter:
                # select all the true negative samples where its score >= positive sample
                ranking = F.asnumpy(F.sum(masked_select(neg_scores[i] >= pos_scores[i], mask[i]), dim=0) + 1)
            else:
                ranking = F.asnumpy(F.sum(neg_scores[i] >= pos_scores[i], dim=0) + 1)
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                'HITS@10': 1.0 if ranking <= 10 else 0.0
            })

    def forward_test_wikikg(self, query, ans, candidate, mode, logs, gpu_id=-1):
        """Do the forward and generate ranking results.

        Parameters
        ----------
        query : Tensor
            input head and relation for test or valid
        ans : Tenseor
            the correct tail entity index
        cadidate : Tensor
            negative sampled tail entity
        """
        scores = self.predict_score_wikikg(query, candidate, mode, to_device=cuda, gpu_id=gpu_id, trace=False)
        if mode == "Valid":
            batch_size = query.shape[0]
            neg_scores = reshape(scores, batch_size, -1)
            for i in range(batch_size):
                ranking = F.asnumpy(F.sum(neg_scores[i] >= neg_scores[i][ans[i]], dim=0) + 1)
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })
        else:
            argsort = F.argsort(scores, dim=1, descending=True)
            logs.append(argsort[:,:10])

    def predict_score_wikikg(self, query, candidate, mode, to_device=None, gpu_id=-1, trace=False):
        num_chunks = len(query)
        chunk_size = 1
        neg_sample_size = candidate.shape[1]
        neg_tail = self.entity_emb(candidate.view(-1), gpu_id, False)
        head = self.entity_emb(query[:,0], gpu_id, False)
        rel = self.relation_emb(query[:,1], gpu_id, False)
        neg_score = self.tail_neg_score(head, rel, neg_tail,
                                        num_chunks, chunk_size, neg_sample_size)
        return neg_score.squeeze()


#     @profile
#     def forward(self, walk_g, pos_g, neg_g, gpu_id=-1):
#         """Do the forward.

#         Parameters
#         ----------
#         walk_g : DGLGraph
#             Graph holding random walk constructed edges.
#         pos_g : DGLGraph
#             Graph holding positive edges.
#         neg_g : DGLGraph
#             Graph holding negative edges.
#         gpu_id : int
#             Which gpu to accelerate the calculation. if -1 is provided, cpu is used.

#         Returns
#         -------
#         tensor
#             loss value
#         dict
#             loss info
#         """
#         pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
# #         pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)
# #         walk_g.ndata['emb'] = self.entity_emb(walk_g.ndata['id'], gpu_id, True)
        
#         # We compose relation embeddings when weight_sharing is composition (only compatible with TransE and RotatE)
#         if self.args.weight_sharing == 'composition':
#             maxlen = walk_g.edata['direction_path'].size(1)

#             mask_arange = th.arange(maxlen)[None, :]

#             mask = mask_arange < walk_g.edata['path_len'][:, None]

#             path_size = walk_g.edata['relation_path'].size()
#             relation_emb = self.relation_emb(walk_g.edata['relation_path'].view(-1), gpu_id, True, mask.view(-1)).view(path_size[0], path_size[1], -1)
#             walk_g.edata['emb'] = self.score_func.prepare_path_embedding(relation_emb, walk_g.edata['direction_path'], walk_g.edata['path_len'], gpu_id)
#         else:
#             pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)

        
#         self.score_func.prepare(pos_g, gpu_id, True)
#         pos_score = self.predict_score(pos_g)
#         if gpu_id >= 0:
#             neg_score = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
#                                                gpu_id=gpu_id, trace=True,
#                                                neg_deg_sample=self.args.neg_deg_sample)
#         else:
#             neg_score = self.predict_neg_score(pos_g, neg_g, trace=True,
#                                                neg_deg_sample=self.args.neg_deg_sample)

#         neg_score = reshape(neg_score, -1, neg_g.neg_sample_size)
# #         print(pos_score, neg_score, flush=True)
#         # subsampling weight
#         # TODO: add subsampling to new sampler
#         #if self.args.non_uni_weight:
#         #    subsampling_weight = pos_g.edata['weight']
#         #    pos_score = (pos_score * subsampling_weight).sum() / subsampling_weight.sum()
#         #    neg_score = (neg_score * subsampling_weight).sum() / subsampling_weight.sum()
#         #else:
#         edge_weight = F.copy_to(pos_g.edata['impts'], get_dev(gpu_id)) if 'impts' in pos_g.edata else None
#         loss, log = self.loss_gen.get_total_loss(pos_score, neg_score, edge_weight)
#         # regularization: TODO(zihao)
#         #TODO: only reg ent&rel embeddings. other params to be added.
# #         if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
# #             coef, nm = self.args.regularization_coef, self.args.regularization_norm
# #             reg = coef * (norm(pos_g.ndata['emb'], nm) + norm(relation_emb[mask], nm))
# # #             print(coef, nm, norm(self.entity_emb.curr_emb(), nm), norm(self.relation_emb.curr_emb(), nm))
# #             log['regularization'] = get_scalar(reg)
# #             loss = loss + reg

            
#         return loss, log
    # @profile
    def forward(self, walk_g, pos_g, neg_g, gpu_id=-1):
        
        """Do the forward.

        Parameters
        ----------
        walk_g : DGLGraph
            Graph holding random walk constructed edges.
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.

        Returns
        -------
        tensor
            loss value
        dict
            loss info
        """
        pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
        walk_g.ndata['emb'] = self.entity_emb(walk_g.ndata['id'], gpu_id, True)
        
        # Use RNN to compose relation embeddings
        if self.args.weight_sharing == 'rnn':
            filler_ones = th.ones(pos_g.number_of_edges(), dtype=th.int8)
            if gpu_id >= 0:
                filler_ones = filler_ones.cuda(gpu_id)
            pos_g.edata['emb'] = self.composition_model(self.relation_emb(pos_g.edata['id'], gpu_id, True).unsqueeze(1), filler_ones, filler_ones.unsqueeze(1), gpu_id, True)
        else:
            pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)
            
            
        
        maxlen = walk_g.edata['direction_path'].size(1)
        mask_arange = th.arange(maxlen)[None, :]
        mask = mask_arange < walk_g.edata['path_len'][:, None]
        mask[(walk_g.edata['impts'] < 0.0001), :] = False
        
        
        # We compose relation embeddings using rnn
        if self.args.weight_sharing == 'rnn':
            path_size = walk_g.edata['relation_path'].size()
            
            if gpu_id >= 0:
                walk_g.edata['direction_path'] = walk_g.edata['direction_path'].cuda(gpu_id)
            relation_emb = self.relation_emb(walk_g.edata['relation_path'].view(-1), gpu_id, True, mask.view(-1)).view(path_size[0], path_size[1], -1)
            walk_g.edata['emb'] = self.composition_model(relation_emb, walk_g.edata['path_len'], walk_g.edata['direction_path'], gpu_id, True)
            
        # We compose relation embeddings using the model bias (only compatible with TransE and RotatE)
        elif self.args.weight_sharing == 'model':
            path_size = walk_g.edata['relation_path'].size()
            expanded_impts = walk_g.edata['impts'].unsqueeze(-1).expand(path_size[0], path_size[1]).contiguous().view(-1)
            relation_emb = self.relation_emb(walk_g.edata['relation_path'].view(-1), gpu_id, True, mask.view(-1)).view(path_size[0], path_size[1], -1)
            walk_g.edata['emb'] = self.score_func.prepare_path_embedding(relation_emb, walk_g.edata['direction_path'], walk_g.edata['path_len'], gpu_id)
        else:
            walk_g.edata['emb'] = self.relation_emb(walk_g.edata['id'], gpu_id, True)

        
        self.score_func.prepare(walk_g, gpu_id, True)
        walk_score = self.predict_score(walk_g)
        self.score_func.prepare(pos_g, gpu_id, True)
        pos_score = self.predict_score(pos_g)
        
        
        edge_weight = th.cat((th.ones(len(pos_score)).to(pos_score.device), walk_g.edata['impts'].to(pos_score.device)))
#         edge_weight = th.cat((th.ones(len(pos_score)).to(pos_score.device), th.zeros(len(walk_g.edata['impts'])).to(pos_score.device)))
#         edge_weight = th.cat((th.zeros(len(pos_score)).to(pos_score.device), walk_g.edata['impts'].to(pos_score.device)))
        pos_score = th.cat((pos_score, walk_score))
#         if gpu_id >= 0:
#             neg_score = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
#                                                gpu_id=gpu_id, trace=True,
#                                                neg_deg_sample=self.args.neg_deg_sample)
#         else:
#             neg_score = self.predict_neg_score(pos_g, neg_g, trace=True,
#                                                neg_deg_sample=self.args.neg_deg_sample)

#         neg_score = reshape(neg_score, -1, neg_g.neg_sample_size)
        if gpu_id >= 0:
            neg_score_1 = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                               gpu_id=gpu_id, trace=True,
                                               neg_deg_sample=self.args.neg_deg_sample)
            neg_score_2 = self.predict_neg_score(walk_g, neg_g, to_device=cuda,
                                               gpu_id=gpu_id, trace=True,
                                               neg_deg_sample=self.args.neg_deg_sample)
        else:
            neg_score_1 = self.predict_neg_score(pos_g, neg_g, trace=True,
                                               neg_deg_sample=self.args.neg_deg_sample)
            neg_score_2 = self.predict_neg_score(walk_g, neg_g, trace=True,
                                               neg_deg_sample=self.args.neg_deg_sample)

        neg_score_1 = reshape(neg_score_1, -1, neg_g.neg_sample_size)
        neg_score_2 = reshape(neg_score_2, -1, neg_g.neg_sample_size)
        
        neg_score = th.cat((neg_score_1, neg_score_2))
#         print(pos_score, neg_score, flush=True)
        # subsampling weight
        # TODO: add subsampling to new sampler
        #if self.args.non_uni_weight:
        #    subsampling_weight = pos_g.edata['weight']
        #    pos_score = (pos_score * subsampling_weight).sum() / subsampling_weight.sum()
        #    neg_score = (neg_score * subsampling_weight).sum() / subsampling_weight.sum()
        #else:
#         edge_weight = F.copy_to(walk_g.edata['impts'], get_dev(gpu_id)) if 'impts' in walk_g.edata else None
        loss, log = self.loss_gen.get_total_loss(pos_score, neg_score, edge_weight)
        # regularization: TODO(zihao)
        #TODO: only reg ent&rel embeddings. other params to be added.
#         if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0:
#             coef, nm = self.args.regularization_coef, self.args.regularization_norm
#             reg = coef * (norm(pos_g.ndata['emb'], nm) + norm(relation_emb[mask], nm))
# #             print(coef, nm, norm(self.entity_emb.curr_emb(), nm), norm(self.relation_emb.curr_emb(), nm))
#             log['regularization'] = get_scalar(reg)
#             loss = loss + reg

            
        return loss, log

    def update(self, gpu_id=-1):
        """ Update the embeddings in the model

        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        self.entity_emb.update(self.args.regularization_coef, self.args.regularization_norm,gpu_id)
        self.relation_emb.update(self.args.regularization_coef, self.args.regularization_norm, gpu_id)
        self.score_func.update(self.args.regularization_coef, self.args.regularization_norm, gpu_id)
        if self.args.weight_sharing == 'rnn':
            self.composition_model.update(self.args.regularization_coef, self.args.regularization_norm, gpu_id)

    def prepare_relation(self, device=None):
        """ Prepare relation embeddings in multi-process multi-gpu training model.

        device : th.device
            Which device (GPU) to put relation embeddings in.
        """
        self.relation_emb = ExternalEmbedding(self.args, self.n_relations, self.rel_dim, device)
        self.relation_emb.init(self.emb_init)
        if self.model_name == 'TransR':
            local_projection_emb = ExternalEmbedding(self.args, self.n_relations,
                                                    self.entity_dim * self.rel_dim, device)
            self.score_func.prepare_local_emb(local_projection_emb)
            self.score_func.reset_parameters()

    def load_relation(self, device=None):
        """ Sync global relation embeddings into local relation embeddings.
        Used in multi-process multi-gpu training model.

        device : th.device
            Which device (GPU) to put relation embeddings in.
        """
        self.relation_emb = ExternalEmbedding(self.args, self.n_relations, self.rel_dim, device)
        self.relation_emb.emb = F.copy_to(self.global_relation_emb.emb, device)
        if self.model_name == 'TransR':
            local_projection_emb = ExternalEmbedding(self.args, self.n_relations,
                                                     self.entity_dim * self.rel_dim, device)
            self.score_func.load_local_emb(local_projection_emb)

    def create_async_update(self):
        """Set up the async update for entity embedding.
        """
        self.entity_emb.create_async_update()

    def finish_async_update(self):
        """Terminate the async update for entity embedding.
        """
        self.entity_emb.finish_async_update()