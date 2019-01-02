#!/usr/bin/env python

'''Make electron density volume from PDB file and configuration parameters'''

import sys
import os
import logging
from py_src import read_config
from py_src import process_pdb
from py_src import py_utils

def main():
    '''Parse command line arguments and generate electron density volume with config file'''
    logging.basicConfig(filename="recon.log", level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    parser = py_utils.MyArgparser(description="make electron density")
    args = parser.special_parse_args()
    logging.info("\n\nStarting.... make_densities")
    logging.info(' '.join(sys.argv))

    try:
        pdb_file = os.path.join(args.main_dir,
                                read_config.get_filename(args.config_file,
                                                         'make_densities',
                                                         'in_pdb_file'))
        pdb_code = None
    except read_config.configparser.NoOptionError:
        pdb_code = read_config.get_filename(args.config_file, 'make_densities', 'pdb_code')
        pdb_file = 'aux/%s.pdb' % pdb_code.upper()
    try:
        num_threads = int(read_config.get_param(args.config_file, 'make_densities', 'num_threads'))
    except read_config.configparser.NoOptionError:
        num_threads = 4
    aux_dir = os.path.join(args.main_dir,
                           read_config.get_filename(args.config_file,
                                                    'make_densities',
                                                    'scatt_dir'))
    den_file = os.path.join(args.main_dir,
                            read_config.get_filename(args.config_file,
                                                     'make_densities',
                                                     'out_density_file'))
    if args.yes:
        to_write = True
    else:
        to_write = py_utils.check_to_overwrite(den_file)

    if to_write:
        timer = py_utils.MyTimer()
        pm = read_config.get_detector_config(args.config_file, show=args.vb) # pylint: disable=C0103
        q_pm = read_config.compute_q_params(pm['detd'], pm['dets_x'],
                                            pm['dets_y'], pm['pixsize'],
                                            pm['wavelength'], pm['ewald_rad'], show=args.vb)
        timer.reset_and_report("Reading experiment parameters") if args.vb else timer.reset()

        if pdb_code is not None:
            process_pdb.fetch_pdb(pdb_code)
        all_atoms = process_pdb.process(pdb_file, aux_dir, pm['wavelength'])
        timer.reset_and_report("Reading PDB") if args.vb else timer.reset()

        den = process_pdb.atoms_to_density_map(all_atoms, q_pm['half_p_res'])
        lp_den = process_pdb.low_pass_filter_density_map(den, threads=num_threads)
        timer.reset_and_report("Creating density map") if args.vb else timer.reset()

        py_utils.write_density(den_file, lp_den, binary=True)
        timer.reset_and_report("Writing densities to file") if args.vb else timer.reset()

        timer.report_time_since_beginning() if args.vb else timer.reset()

if __name__ == "__main__":
    main()
