import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class SummaryChecks(unittest.TestCase):
    def test_smoke_records_are_excluded_from_csv_and_run_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runs = root / 'runs'
            records = []
            for seed in range(5):
                record = dict(dataset='toy', backbone='gcn', method='TGRASP', sigma=1,
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
                writer=csv.DictWriter(f,fieldnames=['dataset','backbone','method','sigma','seed','raw_auc','accuracy'])
                writer.writeheader()
                for seed,auc in enumerate([.2,.8]):
                    writer.writerow(dict(dataset='toy',backbone='gcn',method='TGRASP',sigma=1,seed=seed,raw_auc=auc,accuracy=.9))
            subprocess.run([sys.executable,str(ROOT/'summarize.py'),'--input',str(src),'--output',str(out)],check=True)
            with out.open(newline='') as f: row=next(csv.DictReader(f))
            self.assertEqual(float(row['auc_sep_mean_pct']),80.)
            self.assertEqual(row['complete_five_seed'],'False')


if __name__=='__main__': unittest.main()
