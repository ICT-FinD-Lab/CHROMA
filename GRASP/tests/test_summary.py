import csv
import json
from argparse import Namespace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import make_plan
from run_protocol import protocol_digest


class SummaryChecks(unittest.TestCase):
    def test_smoke_records_are_excluded_from_csv_and_run_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = root / 'runs'
            records = []
            for seed in range(5):
                record = dict(dataset='toy', backbone='gcn', method='CHROMA', sigma=1,
                              seed=seed, raw_auc=.6, accuracy=.8, smoke=seed != 0)
                records.append(record)
                run = runs / str(seed)
                run.mkdir(parents=True)
                (run / 'result.json').write_text(json.dumps(record), encoding='utf-8')

            source = root / 'runs.csv'
            with source.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0].keys())
                writer.writeheader()
                for record, flag in zip(records, ['0', ' True ', '1', 'true', 'TRUE']):
                    writer.writerow(dict(record, smoke=flag))

            for input_path in (runs, source):
                with self.subTest(source=input_path.name):
                    output = root / (input_path.name + '-summary.csv')
                    result = subprocess.run([sys.executable, str(ROOT / 'summarize.py'),
                                             '--input', str(input_path), '--output', str(output)],
                                            check=True, capture_output=True, text=True)
                    self.assertEqual(result.stdout.strip(), 'Formal runs: 1. Cells: 1. Incomplete cells: 1.')
                    with output.open(newline='') as handle:
                        row, = csv.DictReader(handle)
                    self.assertEqual((row['n'], row['seeds'], row['complete_five_seed']),
                                     ('1', '0', 'False'))

    def test_ambiguous_or_smoke_only_csv_does_not_produce_a_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'runs.csv', root / 'summary.csv'
            for flag, error in [('true', 'No formal results'), ('unknown', 'Invalid smoke flag')]:
                with self.subTest(flag=flag):
                    source.write_text('dataset,backbone,method,sigma,seed,raw_auc,accuracy,smoke\n'
                                      f'toy,gcn,GRASP,1,0,0.6,0.8,{flag}\n', encoding='utf-8')
                    result = subprocess.run([sys.executable, str(ROOT / 'summarize.py'),
                                             '--input', str(source), '--output', str(output)],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertFalse(output.exists())

    def test_orientation_is_applied_before_averaging(self):
        with tempfile.TemporaryDirectory() as folder:
            src=Path(folder)/'runs.csv';out=Path(folder)/'summary.csv'
            with src.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['dataset','backbone','method','sigma','seed','raw_auc','accuracy','smoke','protocol_sha256'])
                writer.writeheader()
                for seed,auc in enumerate([.2,.8]):
                    writer.writerow(dict(dataset='toy',backbone='gcn',method='CHROMA',sigma=1,seed=seed,raw_auc=auc,accuracy=.9,
                                         smoke=False,protocol_sha256='a'*64))
            subprocess.run([sys.executable,str(ROOT/'summarize.py'),'--input',str(src),'--output',str(out)],check=True)
            with out.open(newline='') as f: row=next(csv.DictReader(f))
            self.assertEqual(float(row['auc_sep_mean_pct']),80.)
            self.assertEqual(row['complete_five_seed'],'False')

    def test_existing_run_directories_use_their_recorded_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in range(5):
                run = root / str(seed)
                run.mkdir()
                plan = make_plan(Namespace(dataset='cora', backbone='gcn', method='CHROMA', sigma=1.,
                                           seed=seed, device=str(seed % 2), data_dir=run/'data',
                                           groups=run/'groups', output=run, smoke=False))
                plan.update(group_sha256='a'*64, code_sha256={'train.py': 'b'*64})
                (run/'launch.json').write_text(json.dumps(plan))
                row = {k: plan[k] for k in ('dataset','backbone','method','sigma','seed','smoke','epochs','attack_backend')}
                (run/'result.json').write_text(json.dumps(dict(row, raw_auc=.6, accuracy=.8)))

            output = root/'summary.csv'
            cmd = [sys.executable, str(ROOT/'summarize.py'), '--input', str(root), '--output', str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            with output.open(newline='') as f:
                self.assertEqual(next(csv.DictReader(f))['complete_five_seed'], 'True')
            output.unlink()

            last = root/'4'/'launch.json'
            original = json.loads(last.read_text())
            for changes in ({'group_sha256': 'c'*64}, {'epochs': 200}, {'attack_backend': 'sampled'},
                            {'code_sha256': {'train.py':'c'*64}}, {'fixed_split_sha256':'c'*64},
                            {'chroma_config_sha256': 'c'*64},
                            {'training_args': original['training_args'] + ['--pre_ln']}):
                with self.subTest(changes=list(changes)):
                    last.write_text(json.dumps(dict(original, **changes)))
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertRegex(result.stderr, 'Mixed experiment settings|differs between result')
                    self.assertFalse(output.exists())

            last.write_text(json.dumps(dict(original, protocol_sha256='f'*64)))
            rejected = subprocess.run(cmd, capture_output=True, text=True)
            self.assertIn('Protocol ID disagrees', rejected.stderr)
            self.assertFalse(output.exists())
            last.write_text(json.dumps(dict(original, protocol_sha256=protocol_digest(original))))
            subprocess.run(cmd, check=True, capture_output=True)

    def test_csv_rejects_unverified_or_mixed_five_seed_cells(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/'results.csv', Path(temp)/'summary.csv'
            for field, changed, message in [('protocol_sha256', '', 'without a verified protocol'),
                                            ('protocol_sha256', 'b'*64, 'Mixed experiment settings'),
                                            ('group_sha256', 'c'*64, 'Mixed experiment settings'),
                                            ('epochs', 500, 'Mixed experiment settings'),
                                            ('attack_backend', 'sampled', 'Mixed experiment settings')]:
                with self.subTest(field=field, changed=changed):
                    rows = [dict(dataset='cora', backbone='gcn', method='CHROMA', sigma=1, seed=i,
                                 raw_auc=.6 if i < 2 else .9, accuracy=.8, smoke=False, protocol_sha256='a'*64,
                                 group_sha256='a'*64, epochs=3000, attack_backend='dense_exact') for i in range(5)]
                    for row in rows[2:]:
                        row[field] = changed
                    with source.open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=rows[0])
                        writer.writeheader()
                        writer.writerows(rows)
                    result = subprocess.run([sys.executable, str(ROOT/'summarize.py'), '--input', str(source),
                                             '--output', str(output)], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(message, result.stderr)
                    self.assertFalse(output.exists())

    def test_reference_csv_is_recognized_by_content(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'renamed.csv'
            output = Path(temp)/'summary.csv'
            contents = (ROOT/'results/main_table_per_seed.csv').read_bytes()
            source.write_bytes(contents)
            cmd = [sys.executable, str(ROOT/'summarize.py'), '--input', str(source), '--output', str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            with output.open(newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 234)
            self.assertTrue(all(row['complete_five_seed'] == 'True' for row in rows))
            self.assertEqual({row['method'] for row in rows}, {'CLEAN', 'GRASP', 'CHROMA', 'USP'})
            output.unlink()
            source.write_bytes(contents + b'\n')
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertIn('without a verified protocol', result.stderr)
            self.assertFalse(output.exists())

    def test_archived_launch_keeps_its_digest_and_displays_chroma(self):
        recorded_id = 'da52b90e45736087c1f480f816c49d1a73d33db93ceb9e90370f841fb2e425f3'
        launch = dict(dataset='cora', backbone='gcn', method='TGRASP', sigma=1, seed=2, smoke=False,
                      epochs=3000, attack_backend='dense_exact', profile={'epochs': 3000},
                      training_args=['--seed', '2', '--defense_group_algorithm', 'tgrasp'],
                      device='0', code_sha256={'train.py': 'a'*64}, group_sha256='b'*64,
                      tgrasp_config_sha256='c'*64, protocol_sha256=recorded_id)
        self.assertEqual(protocol_digest(launch), recorded_id)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'launch.json').write_text(json.dumps(launch))
            row = {k: launch[k] for k in ('dataset','backbone','method','sigma','seed','smoke','protocol_sha256')}
            (root/'result.json').write_text(json.dumps(dict(row, raw_auc=.6, accuracy=.8)))
            output = root/'summary.csv'
            subprocess.run([sys.executable, str(ROOT/'summarize.py'), '--input', str(root),
                            '--output', str(output)], check=True, capture_output=True)
            with output.open(newline='') as f:
                summary, = csv.DictReader(f)
            self.assertEqual((summary['method'], summary['seeds']), ('CHROMA', '2'))

    def test_old_and_new_method_names_cannot_double_count_a_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/'runs.csv', Path(temp)/'summary.csv'
            header = 'dataset,backbone,method,sigma,seed,raw_auc,accuracy,smoke,protocol_sha256\n'
            row = 'cora,gcn,{},1,2,0.6,0.8,False,' + 'a'*64 + '\n'
            source.write_text(header + row.format('TGRASP') + row.format('CHROMA'))
            result = subprocess.run([sys.executable, str(ROOT/'summarize.py'), '--input', str(source),
                                     '--output', str(output)], capture_output=True, text=True)
            self.assertIn('Duplicate seed', result.stderr)
            self.assertFalse(output.exists())

    def test_exported_five_seeds_keep_the_protocol_and_smoke_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/'export.csv', Path(temp)/'summary.csv'
            fields = ['dataset','backbone','method','sigma','seed','raw_auc','accuracy','protocol_sha256','smoke']
            records = [dict(zip(fields, ['toy','gcn','GRASP',1,seed,.6,.8,'a'*64,False])) for seed in range(5)]
            cmd = [sys.executable, str(ROOT/'summarize.py'), '--input', str(source), '--output', str(output)]
            for names, expected_error in ((fields, None), (fields[:-1], 'must retain the smoke flag'),
                                          (fields[:-2], 'without a verified protocol')):
                with source.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=names, extrasaction='ignore')
                    writer.writeheader()
                    writer.writerows(records)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if expected_error:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected_error, result.stderr)
                    self.assertFalse(output.exists())
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    with output.open(newline='') as handle:
                        self.assertEqual(next(csv.DictReader(handle))['complete_five_seed'], 'True')
                    output.unlink()


if __name__=='__main__': unittest.main()
