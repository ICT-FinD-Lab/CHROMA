"""Exercise the training entry point with a small, offline Planetoid fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.selection import validate_group_partition


class CustomGroupChecks(unittest.TestCase):
    def test_group_ids_are_checked_before_conversion_or_allocation(self):
        for group in ([-1, 0], [0, 10**12], [[0, 1]], [True, False], [0., 1.], []):
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_group_partition(torch.tensor(group), 2)
        group = validate_group_partition(torch.tensor([0, 0], dtype=torch.int16), 2)
        self.assertTrue(torch.equal(group, torch.zeros(2, dtype=torch.long)))

    def test_training_requires_one_integer_group_id_per_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root/'data'/'Planetoid'/'cora'
            (cache/'processed').mkdir(parents=True)
            (cache/'raw').mkdir()
            # Raw filenames suppress downloads. The loader reads the processed fixture.
            for name in ('x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index'):
                (cache/'raw'/f'ind.cora.{name}').touch()
            nodes = torch.arange(20)
            graph = Data(x=torch.eye(len(nodes)), y=nodes % 2,
                         edge_index=torch.stack((nodes, nodes.roll(1))),
                         train_mask=nodes < 12, val_mask=(nodes >= 12) & (nodes < 16),
                         test_mask=nodes >= 16)
            InMemoryDataset.save([graph], str(cache/'processed'/'data.pt'))
            for case, values, error in (
                    ('short', [0], 'Expected one group ID per node (20), got 1'),
                    ('fractional', np.resize([-.25, 1.75], 20), 'group IDs must be integers'),
                    ('valid', np.array(nodes % 3, dtype='>i4'), None)):
                with self.subTest(case=case):
                    output = root/case
                    np.save(root/'cora.npy', values, allow_pickle=False)
                    run = subprocess.run([sys.executable, str(ROOT/'main.py'), '--method', 'TGRASP',
                                          '--device', 'cpu', '--smoke', '--data-dir', str(root/'data'),
                                          '--groups', str(root), '--output', str(output)],
                                         capture_output=True, text=True)
                    launch_file, = output.glob('*/launch.json')
                    result_file = launch_file.with_name('result.json')
                    if error:
                        self.assertNotEqual(run.returncode, 0)
                        self.assertIn(error, run.stderr)
                        self.assertFalse(result_file.exists())
                    else:
                        self.assertEqual(run.returncode, 0, run.stderr)
                        result = json.loads(result_file.read_text())
                        launch = json.loads(launch_file.read_text())
                        self.assertEqual(result['protocol_sha256'], launch['protocol_sha256'])
                        self.assertEqual(result['group_sha256'], launch['group_sha256'])
                        self.assertTrue(0 <= result['raw_auc'] <= 1)
