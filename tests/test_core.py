"""Small CPU checks of result-affecting invariants; no dataset download or training."""
from pathlib import Path
import sys
import unittest
import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import get_noise
from exact_auc import exact_auc
from coloring.initialization import independent_set_coloring
from coloring.initialization import build_initial_partition
from coloring.merging import generate_soft_path,select_structural_knee
from coloring.metrics import assert_proper_coloring


class CoreChecks(unittest.TestCase):
    def test_shared_and_independent_noise(self):
        x=torch.zeros(6,24);group=torch.tensor([0,0,1,1,2,2])
        torch.manual_seed(13)
        noise=get_noise(x,sigma=1,group=group)
        self.assertTrue(torch.equal(noise[0],noise[1]))
        self.assertFalse(torch.equal(noise[0],noise[2]))
        identical=get_noise(x,sigma=1,k=1.5,p=1)
        self.assertTrue(torch.equal(identical,identical[0].expand_as(identical)))
        self.assertEqual(float(get_noise(x,sigma=0,group=group).abs().sum()),0)

    def test_exact_auc_matches_dense_with_ties_and_diagonal(self):
        torch.set_num_threads(2)
        edge=torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]])
        for tied in [False,True]:
            torch.manual_seed(7)
            h=torch.ones(7,5) if tied else torch.randn(7,5)
            z=F.normalize(h-h.mean(dim=1,keepdim=True),p=2,dim=1)
            scores=(z@z.T).relu();scores.fill_diagonal_(0)
            labels=torch.zeros(7,7,dtype=torch.long);labels[edge[0],edge[1]]=1
            expected=float(BinaryAUROC()(scores.reshape(-1),labels.reshape(-1)))
            actual=exact_auc(h,edge,block=3,backend='search')['raw_auc']
            self.assertAlmostEqual(expected,actual,places=6)

    def test_partition_construction(self):
        edge=torch.tensor([[0,1,1,2,2,3,3,4,4,5],[1,0,2,1,3,2,4,3,5,4]])
        coloring=independent_set_coloring(edge,8)
        assert_proper_coloring(edge,8,coloring)
        initial=build_initial_partition(edge,8,coloring,hard_max_group_ratio=.7).group
        states=generate_soft_path(edge,8,initial,global_conflict_cap=.08,local_conflict_p95_cap=.35,
                                  max_group_ratio_cap=.6,audit_top_b=128,lambda_local=.5)
        chosen=select_structural_knee(states,risk_weight=.5)
        self.assertEqual(chosen.group.numel(),8)
        self.assertLessEqual(chosen.metrics['edge_conflict_ratio'],.08)
        self.assertLessEqual(chosen.metrics['local_conflict_p95'],.35)


if __name__=='__main__': unittest.main()
