#!/usr/bin/env python

'''Render EMC data frames to an MP4 movie without a GUI'''

import sys
import argparse
import types
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from .py_src import read_config
    from .py_src import py_utils
except ImportError:
    import os as _os
    sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from py_src import read_config
    from py_src import py_utils


def _build_reader(args):
    stub = types.SimpleNamespace()
    if args.config_file is not None:
        stub.config_file = args.config_file
        read_config.read_gui_config(stub, 'emc')
    else:
        stub.photons_list = [args.emc_fname]
        stub.det_list = [args.det_fname]
        stub.blacklist = None
    py_utils.gen_det_and_emc(stub, classifier=False, mask=args.mask)
    return stub.emc_reader, getattr(stub, 'blacklist', None)


def _load_scores(args):
    if args.hitscore_file is None:
        return None
    with h5py.File(args.hitscore_file, 'r') as f:
        return f['hitscore'][:]


def _select_indices(reader, blacklist, scores, args):
    total = reader.num_frames

    if args.num_frames is not None:
        pool = np.arange(total)
        if blacklist is not None:
            pool = pool[blacklist == 0]
        if scores is not None and args.hitscore_min is not None:
            pool = pool[scores[pool] >= args.hitscore_min]
            print(f'{len(pool)} frames pass hitscore >= {args.hitscore_min}', flush=True)
        if args.num_frames > len(pool):
            print(f'Warning: requested {args.num_frames} frames but only {len(pool)} available; using all.')
            return np.sort(pool)
        return np.sort(np.random.choice(pool, args.num_frames, replace=False))

    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else total
    pool = np.arange(start, end)
    if scores is not None and args.hitscore_min is not None:
        pool = pool[scores[pool] >= args.hitscore_min]
        print(f'{len(pool)} frames pass hitscore >= {args.hitscore_min}', flush=True)
    return pool


def make_movie(args):
    reader, blacklist = _build_reader(args)
    scores = _load_scores(args)
    indices = _select_indices(reader, blacklist, scores, args)
    n = len(indices)
    print(f'Rendering {n} frames -> {args.output}', flush=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.03)

    first = reader.get_frame(int(indices[0]), zoomed=True, avg=True)
    im = ax.imshow(first.T, vmin=0, vmax=args.vmax,
                   origin='lower', interpolation='none', cmap=args.cmap)
    score_str = f'  score={scores[indices[0]]:.1f}' if scores is not None else ''
    ax.set_title(f'Frame {indices[0]}  ({int(first.sum())} photons{score_str})')

    det0 = reader.flist[0]['det']
    cen = det0.get_assembled_cen(zoomed=True, sym=False)
    ax.plot(cen[0], cen[1], '+', c='lime', markersize=20)

    writer = animation.FFMpegWriter(fps=args.fps)
    with writer.saving(fig, args.output, dpi=100):
        for i, num in enumerate(indices):
            frame = reader.get_frame(int(num), zoomed=True, avg=True)
            im.set_data(frame.T)
            score_str = f'  score={scores[num]:.1f}' if scores is not None else ''
            ax.set_title(f'Frame {num}  ({int(frame.sum())} photons{score_str})')
            writer.grab_frame()
            if (i + 1) % 100 == 0 or (i + 1) == n:
                print(f'  {i+1}/{n} frames written', flush=True)

    plt.close(fig)
    print('Done.')


def main():
    parser = argparse.ArgumentParser(
        description='Render EMC frames to an MP4 movie (no GUI required)')
    parser.add_argument('-c', '--config_file', default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-e', '--emc_fname',
                        help='Path to EMC photons file (requires -d)')
    parser.add_argument('-d', '--det_fname',
                        help='Path to detector file (used with -e)')
    parser.add_argument('-o', '--output', default='frames.mp4',
                        help='Output MP4 path (default: frames.mp4)')
    parser.add_argument('--fps', type=float, default=10,
                        help='Frames per second (default: 10)')
    parser.add_argument('--vmax', type=float, default=10,
                        help='Color scale maximum (default: 10)')
    parser.add_argument('--cmap', default='coolwarm',
                        help='Matplotlib colormap (default: coolwarm)')
    parser.add_argument('-n', '--num_frames', type=int, default=None,
                        help='Render N randomly selected frames instead of all')
    parser.add_argument('--start', type=int, default=None,
                        help='First frame index (default: 0); ignored if -n given')
    parser.add_argument('--end', type=int, default=None,
                        help='Last frame index exclusive (default: all); ignored if -n given')
    parser.add_argument('-M', '--mask', action='store_true', default=False,
                        help='Zero out masked pixels')
    parser.add_argument('--hitscore_file', default=None,
                        help='HDF5 file with hitscore array (dataset key: hitscore)')
    parser.add_argument('--hitscore_min', type=float, default=None,
                        help='Only include frames with hitscore >= this value')
    args = parser.parse_args()

    if args.hitscore_min is not None and args.hitscore_file is None:
        parser.error('--hitscore_min requires --hitscore_file')

    if args.emc_fname is not None:
        if args.det_fname is None:
            parser.error('-d/--det_fname required when using -e/--emc_fname')
        args.config_file = None
    elif not args.config_file:
        parser.error('Provide -c/--config_file or -e/--emc_fname + -d/--det_fname')

    make_movie(args)


if __name__ == '__main__':
    main()
