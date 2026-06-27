#!/usr/bin/env python3
'''
Convert AGIPD1M hit frames from a CXI file to EMC format.

Uses the SPI_HITFINDER source from extra_data to identify hit pulses, then
reads the corresponding frames from a pre-processed CXI file. The 16 detector
modules are stacked in module order (0-15), giving 16 * 512 * 128 = 1,048,576
pixels per frame.

Usage:
    python cxi_to_emc.py <proposal> <run> <output.emc> --hits-only [options]

    <proposal>  EuXFEL proposal number, e.g. 10572
    <run>       Run number, e.g. 20
    <output>    Output EMC or HDF5 file (.emc or .h5)
'''

import sys
import os
import argparse
import logging
import time
import numpy as np
import h5py
from concurrent.futures import ThreadPoolExecutor
# from extra_data import RunDirectory, by_id
from extra_data import open_run, by_id
from extra_data.components import AGIPD1M
try:
    from cfelpyutils.geometry import crystfel_utils
    _HAVE_CFELPYUTILS = True
except ImportError:
    _HAVE_CFELPYUTILS = False

# Allow importing writeemc from the Dragonfly utils tree next to this script
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from py_src import writeemc  # pylint: disable=wrong-import-position

NUM_MODULES = 16
MODULE_SLOW = 512
MODULE_FAST = 128
NUM_PIX = NUM_MODULES * MODULE_SLOW * MODULE_FAST  # 1,048,576

HITFINDER_SRC = 'SPB_DET_AGIPD1M-1/REDU/SPI_HITFINDER:output'
BUNCHPATTERN_SRC = 'SPB_RR_SYS/MDL/BUNCH_PATTERN'
# BUNCHPATTERN_PULSEID_KEY = 'sase1.pulseIds.value'
BUNCHPATTERN_PULSEID_KEY = 'laser.pulseIds'


def compute_pix_maps(geom_file):
    """Return flattened (x, y) pixel coordinate arrays from a CrystFEL geometry file.

    Mirrors _compute_pix_maps_stacked in geomtodet.py so that the centerrad
    selection is identical between both scripts.
    """
    geom = crystfel_utils.load_crystfel_geometry(geom_file)
    panels = geom.detector['panels']
    n_modules = max(p['dim_structure'][1] for p in panels.values()) + 1
    max_ss = max(p['orig_max_ss'] for p in panels.values()) + 1
    max_fs = max(p['orig_max_fs'] for p in panels.values()) + 1
    x_map = np.zeros((n_modules, max_ss, max_fs), dtype=np.float32)
    y_map = np.zeros((n_modules, max_ss, max_fs), dtype=np.float32)
    for panel in panels.values():
        mod = panel['dim_structure'][1]
        ss0, ss1 = panel['orig_min_ss'], panel['orig_max_ss'] + 1
        fs0, fs1 = panel['orig_min_fs'], panel['orig_max_fs'] + 1
        ss_g, fs_g = np.meshgrid(np.arange(ss1 - ss0), np.arange(fs1 - fs0),
                                 indexing='ij')
        x_map[mod, ss0:ss1, fs0:fs1] = (ss_g * panel['ssx'] + fs_g * panel['fsx']
                                         + panel['cnx'])
        y_map[mod, ss0:ss1, fs0:fs1] = (ss_g * panel['ssy'] + fs_g * panel['fsy']
                                         + panel['cny'])
    return x_map.ravel(), y_map.ravel()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert AGIPD1M hit frames from a CXI file to EMC format')
    parser.add_argument('proposal', type=int,
                        help='EuXFEL proposal number, e.g. 10572')
    parser.add_argument('run', type=int,
                        help='Run number, e.g. 20')
    parser.add_argument('output',
                        help='Output file (.emc for binary, .h5 for HDF5)')
    parser.add_argument('--hits-only', action='store_true', default=False,
                        help='Write only frames flagged as hits by SPI_HITFINDER')
    parser.add_argument('--cxi', default=None,
                        help='Path to pre-processed CXI file (used with --hits-only). '
                             'Defaults to <exp_base>/usr/Shared/cxi/<run>.cxi deduced from run files.')
    parser.add_argument('--pulses', default=None,
                        help='Pulse slice to select, e.g. "0:100" or "::2". '
                             'Default: all pulses')
    parser.add_argument('--trains', default=None,
                        help='Train slice to select, e.g. "0:500". '
                             'Default: all trains')
    parser.add_argument('--centerrad', type=float, default=None,
                        help='Only include pixels within this radius (pixels) '
                             'from the beam centre (requires --geom)')
    parser.add_argument('--geom', default=None,
                        help='CrystFEL geometry file (required with --centerrad)')
    parser.add_argument('--chunk-size', type=int, default=500,
                        help='Number of trains to load into memory at once (default: 500)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of threads for parallel sparsification (default: 4)')
    parser.add_argument('--ndark', type=int, default=2,
                        help='Dark pulses between pumped pulses in the bunch pattern (default: 2)')
    parser.add_argument('--pulse-offset', type=int, default=4,
                        help='Pulse ID offset applied to hit pulse IDs before matching '
                             'against BUNCHPATTERN laser pulse IDs (default: 4)')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    return parser.parse_args()


def parse_slice(s):
    """Convert a string like '0:100' or '::2' to a Python slice."""
    if s is None:
        return slice(None)
    parts = s.split(':')
    nums = [int(p) if p else None for p in parts]
    if len(nums) == 1:
        return slice(nums[0], nums[0] + 1)
    if len(nums) == 2:
        return slice(nums[0], nums[1])
    return slice(nums[0], nums[1], nums[2])


def load_hit_indices(run, train_sel):
    """Return (hit_train_ids, hit_pulse_ids, total_pulses, total_hit_pulses, hit_scores).

    hit_train_ids/hit_pulse_ids: per-hit EuXFEL train and pulse IDs, used to
    locate each hit in the CXI file by matching against /entry_1/trainId and
    /entry_1/pulseId rather than relying on flat positional indexing.
    """
    try:
        run[HITFINDER_SRC]
    except Exception:
        logging.warning('Hitfinder source not found')
        return None, None, 0, 0, None

    selected = run.select(HITFINDER_SRC).select_trains(train_sel)
    flags_arr = selected.get_array(HITFINDER_SRC, 'data.hitFlag')
    flags = np.array(flags_arr, dtype=bool)
    train_ids = flags_arr.coords['trainId'].values
    pulse_ids = np.array(selected.get_array(HITFINDER_SRC, 'data.pulseId'))

    hit_train_ids = train_ids[flags]
    hit_pulse_ids = pulse_ids[flags]
    total_pulses = len(flags)
    total_hit_pulses = int(flags.sum())
    try:
        scores = np.array(selected.get_array(HITFINDER_SRC, 'data.hitscore'), dtype=np.float32)
        hit_scores = scores[flags]
    except Exception:
        logging.warning('data.hitscore not available in hitfinder source')
        hit_scores = None
    return hit_train_ids, hit_pulse_ids, total_pulses, total_hit_pulses, hit_scores


def compute_pump_flags(run, hit_train_ids, hit_pulse_ids, ndark, pulse_offset=4):
    """Return uint8 array (1=pumped, 0=dark) for each hit, or None if unavailable."""
    try:
        selected = run.select(BUNCHPATTERN_SRC).select_trains(by_id(hit_train_ids))
        bunch_pulse = selected.get_array(BUNCHPATTERN_SRC, BUNCHPATTERN_PULSEID_KEY)
        bp_train_ids = bunch_pulse.coords['trainId'].values
        bp_values = bunch_pulse.values  # shape (n_trains, max_pulses), zeros = unused
    except Exception:
        logging.warning('BUNCHPATTERN source not available; skipping pump flags')
        return None

    # Build per-train set of laser pulse IDs (nonzero entries)
    corrected_pids = hit_pulse_ids - pulse_offset
    laser_pulses = {}
    for i, tid in enumerate(bp_train_ids):
        nonzero = bp_values[i][bp_values[i] > 0]
        if len(nonzero):
            laser_pulses[int(tid)] = set(nonzero.astype(int))

    flags = np.zeros(len(hit_train_ids), dtype=np.uint8)
    for k, (tid, pid) in enumerate(zip(hit_train_ids, corrected_pids)):
        pids = laser_pulses.get(int(tid))
        if pids is not None and int(pid) in pids:
            flags[k] = 1

    n_pumped = int(flags.sum())
    logging.info('Pump flags: %d/%d hits pumped', n_pumped, len(flags))
    return flags


def _sparsify(f):
    """Return (place_ones, place_multi, count_multi) for a single frame.

    numpy releases the GIL during flatnonzero/indexing, so this runs in
    parallel across threads without GIL contention.
    """
    nz = np.flatnonzero(f)
    vals = f[nz]
    ones_mask = vals == 1
    return nz[ones_mask], nz[~ones_mask], vals[~ones_mask]


def process_loaded(data, train_ids, pulse_ids, hit_map, pix_sel, emcwriter, pool):
    """Select frames, batch-convert, sparsify in parallel, write serially."""
    n_trains, n_pulses = data.shape[0], data.shape[1]

    kept_ti, kept_pi = [], []
    for ti in range(n_trains):
        tid = train_ids[ti]
        if hit_map is not None:
            hit_pids = hit_map.get(int(tid), np.array([], dtype=np.int64))
            if len(hit_pids) == 0:
                continue
            pids_for_train = pulse_ids if pulse_ids.ndim == 1 else pulse_ids[ti]
            keep = np.where(np.isin(pids_for_train, hit_pids))[0]
        else:
            keep = np.arange(n_pulses)
        for pi in keep:
            kept_ti.append(ti)
            kept_pi.append(pi)

    if not kept_ti:
        return 0

    frames = data[kept_ti, kept_pi]              # (n_kept, 16, 512, 128)
    flat = frames.reshape(len(kept_ti), NUM_PIX).astype('int32')
    np.clip(flat, 0, None, out=flat)
    if pix_sel is not None:
        flat = flat[:, pix_sel]

    sparse = list(pool.map(_sparsify, flat))

    for po, pm, cm in sparse:
        emcwriter.write_sparse_frame(po, pm, cm)

    return len(kept_ti)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')
    t_start = time.time()

    run_name = f'r{args.run:04d}'
    logging.info('Opening proposal %d run %d', args.proposal, args.run)
    run_proc = open_run(args.proposal, args.run, data="proc")
    run_raw = open_run(args.proposal, args.run, data="raw")
    det = AGIPD1M(run_proc)

    train_sel = parse_slice(args.trains)
    pulse_sel = parse_slice(args.pulses)

    det = det.select_trains(train_sel)
    total_trains = len(det.train_ids)
    logging.info('Selected %d trains', total_trains)

    if args.hits_only and args.cxi is None:
        try:
            first_file = next(iter(run_raw.files)).filename
            exp_base = os.path.dirname(os.path.dirname(os.path.dirname(first_file)))
            args.cxi = os.path.join(exp_base, 'usr', 'Shared', 'cxi', run_name + '.cxi')
            logging.info('Deduced CXI path: %s', args.cxi)
        except Exception:
            logging.error('Could not deduce CXI path automatically; use --cxi')
            sys.exit(1)

    hit_train_ids = None
    if args.hits_only:
        sys.stderr.write('Loading hit flags...\n')
        sys.stderr.flush()
        hit_train_ids, hit_pulse_ids, total_pulses, total_hit_pulses, hit_scores = load_hit_indices(run_proc, train_sel)
        if hit_train_ids is None:
            logging.error('Hitfinder source not found; cannot use --hits-only')
            sys.exit(1)
        hit_rate = total_hit_pulses / total_pulses * 100 if total_pulses else 0.0
        sys.stderr.write(
            'Hit flags loaded: %d/%d hit pulses (%.2f%% hit rate)\n'
            % (total_hit_pulses, total_pulses, hit_rate))
        sys.stderr.flush()

    pix_sel = None
    num_pix = NUM_PIX
    if args.centerrad is not None:
        if args.geom is None:
            logging.error('--geom is required when --centerrad is specified')
            sys.exit(1)
        if not _HAVE_CFELPYUTILS:
            logging.error('cfelpyutils is required for --centerrad but could not be imported')
            sys.exit(1)
        x, y = compute_pix_maps(args.geom)
        pix_sel = np.sqrt(x*x + y*y) <= args.centerrad
        num_pix = int(pix_sel.sum())
        logging.info('centerrad=%.1f: keeping %d/%d pixels', args.centerrad, num_pix, NUM_PIX)

    ext = os.path.splitext(args.output)[1].lower()
    use_hdf5 = ext != '.emc'

    print('Number of workers: %d' % args.num_workers)
    total_frames = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        if hit_train_ids is not None:
            # Resolve hit (trainId, pulseId) pairs to CXI frame indices
            sys.stderr.write('Loading %d hit frames from %s...\n'
                             % (len(hit_train_ids), args.cxi))
            sys.stderr.flush()
            with h5py.File(args.cxi) as h5:
                cxi_train_ids = h5['/entry_1/trainId'][:]
                cxi_pulse_ids = h5['/entry_1/pulseId'][:]
                cxi_lookup = {(int(t), int(p)): i
                              for i, (t, p) in enumerate(zip(cxi_train_ids, cxi_pulse_ids))}
                hit_indices = []
                valid = np.zeros(len(hit_train_ids), dtype=bool)
                for k, (t, p) in enumerate(zip(hit_train_ids, hit_pulse_ids)):
                    idx = cxi_lookup.get((int(t), int(p)))
                    if idx is not None:
                        hit_indices.append(idx)
                        valid[k] = True
                n_missing = int((~valid).sum())
                if n_missing:
                    logging.warning('%d/%d hits not found in CXI (coverage mismatch); skipping',
                                    n_missing, len(valid))
                    hit_train_ids = hit_train_ids[valid]
                    hit_pulse_ids = hit_pulse_ids[valid]
                    if hit_scores is not None:
                        hit_scores = hit_scores[valid]
                hit_indices = np.array(hit_indices)
                data = h5['/entry_1/instrument_1/detector_1/data'][hit_indices]
            pump_flags = compute_pump_flags(run_raw, hit_train_ids, hit_pulse_ids, args.ndark, args.pulse_offset)
            # data: (n_hits, 16, 512, 128)
            flat = data.reshape(len(hit_indices), NUM_PIX).astype('int32')
            np.clip(flat, 0, None, out=flat)
            if pix_sel is not None:
                flat = flat[:, pix_sel]
            sparse = list(pool.map(_sparsify, flat))
            stem, ext_out = os.path.splitext(args.output)
            pumped_mask = pump_flags.astype(bool) if pump_flags is not None else None
            if pumped_mask is not None:
                pumped_out = stem + '_pumped' + ext_out
                dark_out   = stem + '_dark'   + ext_out
                logging.info('Splitting output: pumped -> %s, dark -> %s', pumped_out, dark_out)
                # Create, write, finish each writer sequentially to avoid PID-based
                # temp file name collision when two EMCWriters share the same output dir.
                writer_pumped = writeemc.EMCWriter(pumped_out, num_pix, hdf5=use_hdf5)
                for i, (po, pm, cm) in enumerate(sparse):
                    if pumped_mask[i]:
                        writer_pumped.write_sparse_frame(po, pm, cm)
                writer_pumped.finish_write()
                writer_dark = writeemc.EMCWriter(dark_out, num_pix, hdf5=use_hdf5)
                for i, (po, pm, cm) in enumerate(sparse):
                    if not pumped_mask[i]:
                        writer_dark.write_sparse_frame(po, pm, cm)
                writer_dark.finish_write()
            else:
                writer = writeemc.EMCWriter(args.output, num_pix, hdf5=use_hdf5)
                for po, pm, cm in sparse:
                    writer.write_sparse_frame(po, pm, cm)
                writer.finish_write()
            total_frames = len(hit_indices)
            out_dir = os.path.dirname(os.path.abspath(args.output))
            if pumped_mask is not None:
                for mask, suffix in [(pumped_mask, '_pumped'), (~pumped_mask, '_dark')]:
                    fname = os.path.join(out_dir, f'{run_name}{suffix}_hitscore.h5')
                    with h5py.File(fname, 'w') as f:
                        if hit_scores is not None:
                            f.create_dataset('hitscore', data=hit_scores[mask])
                        f.create_dataset('train_id', data=hit_train_ids[mask].astype(np.uint64))
                        f.create_dataset('pulse_id', data=hit_pulse_ids[mask].astype(np.uint64))
                    logging.info('Saved hit info (%d entries) to %s', mask.sum(), fname)
            else:
                fname = os.path.join(out_dir, f'{run_name}_hitscore.h5')
                with h5py.File(fname, 'w') as f:
                    if hit_scores is not None:
                        f.create_dataset('hitscore', data=hit_scores)
                    f.create_dataset('train_id', data=hit_train_ids.astype(np.uint64))
                    f.create_dataset('pulse_id', data=hit_pulse_ids.astype(np.uint64))
                logging.info('Saved hit info (%d entries) to %s', len(hit_train_ids), fname)
        else:
            # No hit map: chunk through all trains
            emcwriter = writeemc.EMCWriter(args.output, num_pix, hdf5=use_hdf5)
            all_train_ids = det.train_ids
            chunk_size = args.chunk_size
            n_chunks = (total_trains + chunk_size - 1) // chunk_size
            logging.info('Processing %d trains in %d chunk(s) of %d',
                         total_trains, n_chunks, chunk_size)
            for ci in range(n_chunks):
                chunk_tids = all_train_ids[ci * chunk_size:(ci + 1) * chunk_size]
                sys.stderr.write('Loading chunk %d/%d (%d trains)...\n'
                                 % (ci + 1, n_chunks, len(chunk_tids)))
                sys.stderr.flush()
                chunk_det = det.select_trains(by_id[list(chunk_tids)])
                arr = chunk_det.get_array('image.data', pulses=pulse_sel, unstack_pulses=True)
                # (module, train, pulse, ss, fs) -> (train, pulse, module, ss, fs)
                data = np.moveaxis(arr.values, 0, 2)
                train_ids = arr.coords['train'].values
                pulse_ids = arr.coords['pulse'].values
                total_frames += process_loaded(
                    data, train_ids, pulse_ids, None, pix_sel, emcwriter, pool)
            emcwriter.finish_write()

    sys.stderr.write('Frames written: %d\n' % total_frames)
    elapsed = time.time() - t_start
    fps = total_frames / elapsed if elapsed > 0 else 0.0
    logging.info('Done. Total frames written: %d', total_frames)
    print('Total time: %.1f s' % elapsed)
    print('Frames per second: %.1f' % fps)


if __name__ == '__main__':
    main()
