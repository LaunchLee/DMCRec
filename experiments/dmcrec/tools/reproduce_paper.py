"""Preview or replay frozen selected settings from the revised DMCRec study.

No training or data loading occurs without --execute. This entry point replays
selected settings; it does not repeat the validation search. The reported
seed-level results are preserved separately in the release's research_results.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / 'experiments/dmcrec'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def configurations():
    return json.loads((PROJECT / 'configs/paper_configurations.json').read_text())


def command(args, config):
    """Build the original shared runner command using the recorded constructor parameters."""
    params = config['params']
    native_batch = params['batch_size']
    epochs = 2 if args.smoke else 200
    cmd = [sys.executable, str(PROJECT / 'main.py'),
           '--dataset', config['dataset'], '--models', config['model'],
           '--variants', config['variant'], '--seeds', *map(str, args.seeds),
           '--protocol', 'corrected', '--implementation', 'corrected',
           '--split-source', 'random', '--split-seed', '42',
           '--val-size', '0.1', '--test-size', '0.2', '--setting', 'original',
           '--epochs', str(epochs), '--batch-size', str(native_batch),
           '--embedding-dim', '128', '--semantic-dim', '64',
           '--learning-rate', str(params.get('learning_rate', 0.001)),
           '--device', args.device, '--data-dir', str(args.data_dir),
           '--cache-dir', str(args.output_dir / '_feature_cache'),
           '--output-dir', str(args.output_dir / ('smoke' if args.smoke else 'replication')),
           '--refresh', config['refresh_policy'],
           '--inference', *config['inference'],
           '--cf-references', *map(str, config['reference_grid'])]
    if args.smoke:
        cmd += ['--max-items', '256']
    # Execution identity and epoch overrides are handled by dedicated options.
    excluded = {'name', 'seed', 'use_gpu', 'device', 'implementation', 'n_epochs', 'max_iter'}
    for key, value in params.items():
        if key not in excluded:
            cmd += ['--param', config['model'] + '.' + key + '=' + json.dumps(value)]
    if config['model'] == 'WMF':
        cmd += ['--param', 'WMF.max_iter=' + str(epochs)]
    return cmd


def run_mf(args, config):
    """Replay the independently adapted MF pair, using the recorded selected lr/reference."""
    import torch
    from experiments.dmcrec.main import parse_args, load_data, jsonable, source_hashes
    from experiments.dmcrec.tools.remote import make_macr_model, macr_evaluate
    torch.set_num_threads(1)
    ds, model_name = config['dataset'], config['model']
    data_args = parse_args(['--dataset', ds, '--data-dir', str(args.data_dir),
                           '--cache-dir', str(args.output_dir / '_feature_cache'),
                           '--protocol', 'corrected', '--implementation', 'corrected',
                           '--split-source', 'random', '--split-seed', '42',
                           '--val-size', '0.1', '--test-size', '0.2',
                           *(['--max-items', '256'] if args.smoke else [])])
    split, _ = load_data(data_args, 'original')
    for seed in args.seeds:
        destination = args.output_dir / ('smoke' if args.smoke else 'replication') / ds / model_name / ('seed-' + str(seed))
        if destination.exists():
            raise FileExistsError('Refusing to overwrite existing run: ' + str(destination))
        destination.mkdir(parents=True)
        fitted = make_macr_model(split.train_set.num_users, split.train_set.num_items,
                                 model=model_name, seed=seed, device='cpu')
        fitted.fit(split.train_set, epochs=2 if args.smoke else 200,
                   batch_size=8192, learning_rate=config['selected']['learning_rate'])
        fitted.reference = config['selected']['reference']
        evaluations = {}
        for mode in (['main', 'factual', 'counterfactual'] if model_name == 'MACR-MF' else ['main']):
            fitted.mode = mode
            evaluations[mode] = macr_evaluate(fitted, split.train_set, split.test_set,
                                              exclude_set=split.val_set)['metrics']
        result = dict(dataset=ds, model=model_name, seed=seed, evaluations=evaluations,
                      data_report=split.preprocessing_report, source_sha256=source_hashes(),
                      selected_settings=config, smoke=args.smoke,
                      replay_scope='frozen selected settings; no new validation search',
                      torch_version=torch.__version__, threads=1)
        (destination / 'result.json').write_text(json.dumps(jsonable(result), indent=2))
        print(destination / 'result.json', flush=True)


def main(argv=None):
    configs = configurations()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--group', choices=list(configs))
    parser.add_argument('--data-dir', type=Path, default=PROJECT / 'datasets')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'replication_results')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(configs))
        return
    if not args.group:
        parser.error('--group or --list is required')
    if len(set(args.seeds)) != len(args.seeds) or any(s not in range(42, 47) for s in args.seeds):
        parser.error('Use unique recorded training seeds 42–46')
    config = configs[args.group]
    print(json.dumps(dict(group=args.group, configuration=config,
                          seeds=args.seeds, smoke=args.smoke,
                          execute=args.execute, scope='replay selected settings; no hyperparameter search'), indent=2))
    mf = config['model'] in {'MF-BCE', 'MACR-MF'}
    if mf:
        if args.device != 'cpu':
            parser.error('The recorded MF pair uses the one-thread CPU protocol; use --device cpu')
        if args.execute:
            run_mf(args, config)
    else:
        cmd = command(args, config)
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
