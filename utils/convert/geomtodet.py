#!/usr/bin/env python

'''
Convert CrystFEL geometry file to detector file
Can specify mask file separately.

Needs:
    <geom_fname> - Path to CrystFEL geometry h5 file

Produces:
    Detector file in output_folder
'''

from __future__ import print_function
import sys
import os
import logging
import numpy as np
import h5py
try:
    from six.moves import configparser
except ImportError:
    import configparser
#Add utils directory to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from py_src import py_utils # pylint: disable=wrong-import-position
from py_src import read_config # pylint: disable=wrong-import-position
from py_src import detector # pylint: disable=wrong-import-position
try:
    # from cfelpyutils import crystfel_utils, geometry_utils
    from cfelpyutils.geometry import crystfel_utils, geometry
except ImportError:
    print('Need cfelpyutils package to safely parse geometry file.')
    print('Install from pip if possible.')
    raise

def _compute_pix_maps_stacked(detector_dict):
    """Build (n_modules, ss, fs) pixel maps honouring dim_structure module indices.

    compute_pix_maps from cfelpyutils creates a single (ss, fs) slab, so panels
    from different modules (dim_structure[1] != 0) overwrite each other. This
    function stacks all modules into a 3-D array and returns flattened x, y.
    """
    panels = detector_dict['panels']
    n_modules = max(p['dim_structure'][1] for p in panels.values()) + 1
    max_ss = max(p['orig_max_ss'] for p in panels.values()) + 1
    max_fs = max(p['orig_max_fs'] for p in panels.values()) + 1
    x_map = np.zeros((n_modules, max_ss, max_fs), dtype=np.float32)
    y_map = np.zeros((n_modules, max_ss, max_fs), dtype=np.float32)
    for panel in panels.values():
        mod = panel['dim_structure'][1]
        ss0, ss1 = panel['orig_min_ss'], panel['orig_max_ss'] + 1
        fs0, fs1 = panel['orig_min_fs'], panel['orig_max_fs'] + 1
        ss_g, fs_g = np.meshgrid(np.arange(ss1 - ss0), np.arange(fs1 - fs0), indexing='ij')
        x_map[mod, ss0:ss1, fs0:fs1] = ss_g * panel['ssx'] + fs_g * panel['fsx'] + panel['cnx']
        y_map[mod, ss0:ss1, fs0:fs1] = ss_g * panel['ssy'] + fs_g * panel['fsy'] + panel['cny']
    return x_map.ravel(), y_map.ravel()


def main():
    """Parse command line arguments and convert file"""
    logging.basicConfig(filename='recon.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.MyArgparser(description='cheetahtodet')
    parser.add_argument('geom_fname', help='CrystFEL geometry file to convert to detector format')
    parser.add_argument('-M', '--mask',
                        help='Path to detector style mask (0:good, 1:no_orient, 2:bad) in h5 file')
    parser.add_argument('--mask_dset',
                        help='Data set in mask file. Default: /data/data', default='data/data')
    parser.add_argument('--dragonfly_mask',
                        help='Whether mask has Dragonfly style values or not. (Default: false)',
                        default=False, action='store_true')
    parser.add_argument('--wavelength', type=float,
                        help='X-ray wavelength in Angstroms (overrides config file)')
    parser.add_argument('--detd', type=float,
                        help='Detector distance in mm (overrides config/geometry file)')
    parser.add_argument('--pixsize', type=float,
                        help='Pixel size in mm (overrides config/geometry file)')
    parser.add_argument('--polarization', default='x',
                        help='Polarization direction: x, y, or none (default: x)')
    parser.add_argument('--stoprad', type=float, default=0.,
                        help='Stop radius in pixels (default: 0)')
    parser.add_argument('--output_folder', default='.',
                        help='Output folder for detector file (default: current directory)')
    args = parser.special_parse_args()

    logging.info('Starting cheetahtodet...')
    logging.info(' '.join(sys.argv))

    # Try config file first; fall back to CLI args and geometry file
    pm = {}
    output_folder = args.output_folder
    try:
        pm = read_config.get_detector_config(args.config_file, show=args.vb) # pylint: disable=invalid-name
        output_folder = read_config.get_filename(args.config_file, 'emc', 'output_folder')
    except (configparser.NoSectionError, configparser.NoOptionError,
            configparser.MissingSectionHeaderError):
        pass

    # CLI args override config file values
    if args.wavelength is not None:
        pm['wavelength'] = args.wavelength
    if args.detd is not None:
        pm['detd'] = args.detd
    if args.pixsize is not None:
        pm['pixsize'] = args.pixsize
    pm.setdefault('polarization', args.polarization)
    pm.setdefault('stoprad', args.stoprad)

    # CrystFEL geometry files have coordinates in pixel size units
    geom = crystfel_utils.load_crystfel_geometry(args.geom_fname)

    # Extract wavelength/detd/pixsize from geometry file when not set via config or CLI
    first_panel = next(iter(geom.detector['panels'].values()))
    if 'wavelength' not in pm:
        photon_energy = geom.beam.get('photon_energy')
        if photon_energy:
            pm['wavelength'] = 12398.4 / photon_energy  # eV -> Angstroms
    if 'detd' not in pm:
        pm['detd'] = first_panel['clen'] * 1000.  # metres -> mm
    if 'pixsize' not in pm:
        pm['pixsize'] = 1000. / first_panel['res']  # pixels/m -> mm/pixel

    if 'wavelength' not in pm:
        parser.error('wavelength not found in geometry file or config; pass --wavelength')

    pm.setdefault('ewald_rad', pm['detd'] / pm['pixsize'])

    # dets_x/dets_y not needed for geometry-based conversion; use dummy if absent
    pm.setdefault('dets_x', 1)
    pm.setdefault('dets_y', 1)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'], pm['dets_y'],
                                            pm['pixsize'], pm['wavelength'],
                                            pm['ewald_rad'], show=args.vb)

    x, y = _compute_pix_maps_stacked(geom.detector)
    z = pm['detd'] / pm['pixsize']
    pm['pixsize'] = 1.
    
    det = detector.Detector()
    norm = np.sqrt(x*x + y*y + z*z)
    qscaling = 1. / pm['wavelength'] / q_pm['q_sep']
    det.qx = x * qscaling / norm
    det.qy = y * qscaling / norm
    det.qz = qscaling * (z / norm - 1.)
    det.corr = pm['detd']*(pm['pixsize']*pm['pixsize']) / np.power(norm, 3.0)
    det.corr *= read_config.compute_polarization(pm['polarization'], x, y, norm)
    if args.mask is None:
        radius = np.sqrt(x*x + y*y)
        rmax = min(np.abs(x.max()), np.abs(x.min()), np.abs(y.max()), np.abs(y.min()))
        det.raw_mask = np.zeros(det.corr.shape, dtype='u1')
        det.raw_mask[radius > rmax] = 1
    else:
        with h5py.File(args.mask, 'r') as fptr:
            det.raw_mask = fptr[args.mask_dset][:].astype('u1').flatten()
        if not args.dragonfly_mask:
            det.raw_mask = 2 - 2*det.raw_mask

    det.detd = pm['detd'] / pm['pixsize']
    det.ewald_rad = pm['ewald_rad']
    det_file = output_folder + '/' + os.path.splitext(os.path.basename(args.geom_fname))[0]
    try:
        import h5py
        det_file += '.h5'
    except ImportError:
        det_file += '.dat'
    logging.info('Writing detector file to %s', det_file)
    sys.stderr.write('Writing detector file to %s\n'%det_file)
    det.write(det_file)

if __name__ == '__main__':
    main()
