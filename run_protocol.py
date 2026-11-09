"""Identify comparable training runs across seeds and working directories."""
import hashlib
import json
import re


def canonical_method(name):
    """Use the current method names for CLI input and summary rows."""
    return {'TGRASP': 'CHROMA', 'TGRASP-default': 'CHROMA-default'}.get(name, name)


def check_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Expected a lowercase SHA-256 digest')
    return value


def protocol_digest(launch):
    """Use the same recorded settings for new runs and earlier launch files."""
    required = ('dataset', 'backbone', 'method', 'sigma', 'smoke', 'epochs',
                'attack_backend', 'training_args', 'profile', 'device', 'code_sha256')
    missing = [key for key in required if key not in launch]
    if missing:
        raise ValueError('Incomplete launch settings: ' + ', '.join(missing))
    if not launch['training_args'] or not launch['profile'] or 'train.py' not in launch['code_sha256']:
        raise ValueError('Launch must record training arguments, model profile and source hashes')

    args = iter(launch['training_args'])
    training_args = []
    for arg in args:
        if arg in ('--seed', '--data_dir', '--group_dir', '--device'):
            if next(args, None) is None:
                raise ValueError(f'Missing value for {arg} in launch')
        else:
            training_args.append(arg)

    settings = {key: launch[key] for key in ('dataset', 'backbone', 'method', 'smoke',
                                              'epochs', 'profile', 'attack_backend')}
    settings.update(version=1, sigma=float(launch['sigma']), training_args=training_args,
                    device_type='cpu' if launch['device'] == 'cpu' else 'cuda',
                    fixed_split_sha256=launch.get('fixed_split_sha256'),
                    code_sha256={path: digest for path, digest in launch['code_sha256'].items()
                                 if path not in ('summarize.py', 'run_protocol.py')})
    if canonical_method(launch['method']) == 'CHROMA':
        settings['group_sha256'] = check_digest(launch.get('group_sha256'))
        # The original field name is part of previously recorded protocol IDs.
        config_key = launch['method'].lower() + '_config_sha256'
        settings[config_key] = launch.get(config_key)
    encoded = json.dumps(settings, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()
