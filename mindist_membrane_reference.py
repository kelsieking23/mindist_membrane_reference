import os
import argparse
import mdtraj
import numpy as np 
import time
import sys

__version__ = 1.1
'''
This is a script that is meant to calculate minimum distance matrices between several single-point subtrajectories.
The intended implementation is to calculate distances between:
    1) a 3D array of points corresponding to the center-of-masses (COMs) of N residues over a trajectory (N_frames, N_residues, 3)
    2) a 3D array of points corresponding to the positions of lipid reference atoms, i.e. the P atom of a lipid headgroup (N_frames, N_references, 3)
Theoretically, this script could apply to any two single-point groups of interest over a trajectory, e.g. COM distances between residues, residues &
waters, etc. Since MDtraj couldn't quite do what I wanted to do with its own implementations, I use MDtraj to load coordinates, topology information, 
and unit cell information. Distance calculations are performed with NumPy and use a minimum image convention (thanks chatGPT!). The code was validated
by comparing to gmx distance outputs. See membrane_protien_dist_prototype.ipynb for prototyping/some extra notes/validation. 
The implementation here isn't meant to be universal - please check your outputs to make sure they make sense. 
'''


class Ndx:

    def __init__(self, gro=None, ndx=None, types = {}, ndxt=None):
        self.gro = gro
        self.ndx = ndx
        self.ndxt_groups = ndxt
        self.types = types

    @classmethod
    def from_ndx(cls, ndx_file):
        types = {}
        with open(ndx_file, 'r') as f:
            for line in f:
                if line.startswith('['):
                    group = line.strip('[').strip().strip(']').strip()
                    types[group] = []
                else:
                    types[group] = types[group] + line.strip().split()
        return cls(gro=None, ndx=ndx_file, types=types, ndxt=None)


def compute_pbc_distances_variable_box(reference_atoms, residue_coms, box_sizes):
    """
    Compute distances using the minimum image convention (MIC) using unit cell info
    from the trajectory.

    Parameters:
    - reference_atoms: (N_frames, N_references, 3) array of reference atom positions.
    - residue_coms: (N_frames, N_residues, 3) array of center of mass positions.
    - box_sizes: (N_frames, 3) array of box dimensions per frame.

    Returns:
    - distances: (N_frames, N_n_references, N_residues) array of MIC distances.
    """
    # Expand box_sizes to enable broadcasting over atoms and residues
    box_sizes = box_sizes[:, np.newaxis, np.newaxis, :] 

    # Compute pairwise differences and apply MIC
    delta = reference_atoms[:, :, np.newaxis, :] - residue_coms[:, np.newaxis, :, :]
    delta -= box_sizes * np.round(delta / box_sizes)

    # Compute Euclidean distances
    distances = np.linalg.norm(delta, axis=-1)  # Shape: (N_frames, N_n_references, N_residues)

    return distances

if __name__ == '__main__':
    description = '''This script performs minimum distance calculations between the center-of-mass (COM) or specified atom groups (residues) and a reference group containing membrane atoms.
The inputs (-residue_ndx and -reference_ndx) are GROMACS index files. For -residue_ndx, this index file can contain any amount of atom groups. For each index group, the COM will be computed, and the minimum distance between the COM
and the atoms specified in -reference_ndx will be computed (for each frame of the supplied trajectory). For -reference_ndx: it is reccomended that this index file contains any number of lipid headgroup atoms to speed up the calculation. 
'''
    epilog = '''Example usage:\n

python mindist_membrane_reference.py -f example_files/example_trajectory.xtc -s example_files/example_topology.pdb -residue_ndx example_files/residue_ndx.ndx -reference_ndx example_files/ref_idx_membrane.ndx  
-ref_id POPC -o example_files/example_output.npy --log example_files/example.log
'''
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument('-f', type=str, help='trajectory file (e.g. .xtc, .trr, etc)')
    parser.add_argument('-s', type=str, help='topology (.pdb)')
    parser.add_argument('-residue_ndx', type=str, help='GROMACS index file containing residues (or groups) of interest. can have any amount of groups. for each index group, the COM will be taken, and minimum distance between the COM and the reference index will be calculated. ')
    parser.add_argument('-reference_ndx', type=str, help='GROMACS index files containing reference atoms for the membrane. reccomended: headgroup atoms of the membrane lipids. reference atoms need to be a single index group.')
    parser.add_argument('-ref_id', type=str, default='', help=' (optional) index group name in the reference index file (-reference_ndx) to use. if not specified, will use the first')
    parser.add_argument('--stride', type=int, default=1, help='(optional, default=1) stride (default 1)')
    parser.add_argument('--use_ref_com', action='store_true', help='if specified, will take minimum distance between the residues and the COM of the reference atoms supplied.')
    parser.add_argument('-o', type=str, default='distance.npy', help='output filename (default: distance.npy)')
    parser.add_argument('-l', '--log', default='distance.log', type=str, help='log output file (default: distance.log)')
    parser.add_argument('--quiet', action='store_true', help='quiet (not verbose)')
    parser.add_argument('-v', '--version', action='store_true', help='print version and quit')
    args = parser.parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    # some error handling
    try:
        assert os.path.isfile(args.f)
    except:
        raise FileNotFoundError("No such file or directory: ''{}''".format(args.f))
    try:
        assert os.path.isfile(args.s)
    except:
        raise FileNotFoundError("No such file or directory: ''{}''".format(args.s))
    try:
        assert os.path.isfile(args.residue_ndx)
    except:
        raise FileNotFoundError("No such file or directory: ''{}''".format(args.residue_ndx))
    try:
        assert os.path.isfile(args.reference_ndx)
    except:
        raise FileNotFoundError("No such file or directory: ''{}''".format(args.reference_ndx))
    try:
        assert os.path.isdir(os.path.dirname(args.o))
    except:
        try:
            os.mkdir(os.path.dirname(args.o))
        except:
            raise FileNotFoundError("No such file or directory: '{}'".format(os.path.dirname(args.o)))
    base, ext = os.path.splitext(args.o)  
    if ext != '.npy':
        corrected_out = base + '.npy'
        if not args.quiet:
            print('Warning: changing the output filename from {} to {}'.format(args.o, corrected_out))
        args.o = corrected_out  

    if not args.quiet:
        print('Parameters:')
        print('*************************************************')
        print('Submission dir: {}'.format(os.getcwd()))
        for key, value in args.__dict__.items():
            print(key, ': ', str(value))
        print('*************************************************\n')
    with open(args.log, 'w') as f:
        f.write('Submission dir: {}'.format(os.getcwd()))
        for key, value in args.__dict__.items():
            f.write(key + ': ' + str(value))
    # load trajectory, topology, index
    if not args.quiet:
        print('Loading trajectory and topology...')
    traj = mdtraj.load(args.f, top=args.s, stride=args.stride)
    top = traj.topology
    if not args.quiet:
        print('Trajectory loaded.')
        print(traj)
    if not args.quiet:
        print('Loading index...')
    # loading reference .ndx
    ref_ndx = Ndx.from_ndx(args.reference_ndx)
    if args.ref_id != '':
        ref_idx = [int(i)-1 for i in ref_ndx.types[args.ref_id]]
    else:
        ref_id = list(ref_ndx.types.keys())[0]
        if not args.quiet:
            print('-ref_id not specified. using {} as the reference group'.format(ref_id))
        ref_idx = [int(i)-1 for i in ref_ndx.types[ref_id]]
    ref_traj = traj.atom_slice(ref_idx)
    if args.use_ref_com:
        reference_atoms = mdtraj.compute_center_of_mass(ref_traj)
    else:
        reference_atoms = ref_traj._xyz

    # loading the residue .ndx
    res_ndx = Ndx.from_ndx(args.residue_ndx)
    residue_coms = []
    for key in res_ndx.types.keys():
        res = traj.atom_slice([int(i)-1 for i in res_ndx.types[key]])
        residue_coms.append(mdtraj.compute_center_of_mass(res))
    residue_coms = np.stack(residue_coms,axis=1)

    if not args.quiet:
        print('Loading indices complete.\n')
        print('Residue COMs shape: {}'.format(residue_coms.shape))
        print('Reference atom shape: {}'.format(reference_atoms.shape))
    if not args.quiet:
        print('Generating distance matrix...')
    start = time.time()
    distances = compute_pbc_distances_variable_box(reference_atoms, residue_coms, traj.unitcell_lengths)
    end = time.time()
    if not args.quiet:
        print('Distance matrix generated. Time elapsed: {:.2f} minutes\n'.format((end-start)/60))
        print('There are {} frames in the trajectory, (axis=0) {} specified residue groups (axis 1), and {} specified reference atoms (axis 2).'.format(distances.shape[0], residue_coms.shape[1], reference_atoms.shape[1]))
        print('Calculating minimum distances with respect to residues (axis=2) over all frames (axis=0) will result in array shape ({}, {})'.format(residue_coms.shape[0], residue_coms.shape[1]))
        print('Finding minima...')
    mindist = np.min(distances, axis=1)
    if not args.quiet:
        print('Mindist calculation complete.')
        print('Writing output file...')
    np.save(args.o, mindist)
    if not args.quiet:
        print('Wrote {}'.format(args.o))
        print('Job complete.')
    





