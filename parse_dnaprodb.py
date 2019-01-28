from __future__ import print_function
import pprint as pp
import argparse
import json
import gzip
import re
import os
import sys


AMINO_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'


PROTEIN_MAPPINGS = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
}


def pfile(m, f):
    f = open(f, 'a+')
    print(m, file=f)


def perr(m):
    # print to stderr
    print(m, file=sys.stderr)


def open_maybe_gzipped(f):
    # open gzipped files
    with open(f, 'rb') as test_read:
        byte1, byte2 = test_read.read(1), test_read.read(1)
        if byte1 and ord(byte1) == 0x1f and byte2 and ord(byte2) == 0x8b:
            f = gzip.open(f, mode='rt')
        else:
            f = open(f, 'rt')
        return f


def parse_arguments():
    # parse_arguments
    parser = argparse.ArgumentParser(description='Specify input file')
    parser.add_argument('infile', type=str, help='Collection file')
    parser.add_argument('outfile', type=str, help='Output file with annotations')
    return parser.parse_args()


def iterate_json(db):
    # generator for list of parsed json entries
    for line in db:
        entry = json.loads(line)
        yield entry


def rename_residues(res):
    # convert 3 letter identifiers to 1 letter identifiers
    for key in res:
        try:
            # change to form {res_id: (res_name (1 letter), index)}
            res[key] = (PROTEIN_MAPPINGS[res[key][0]], res[key][1])
        # chemically  modified bases are booted out
        except KeyError:
            return None
    return res


def get_contacts(chain_identifiers, structure):
    interactions = structure['interactions']['nucleotide-residue_interactions']
    identifier = structure['pdbid']
    interaction_list = dict()

    for key in chain_identifiers:
        interaction_list['.'.join([identifier, key])] = [chain_identifiers[key][1], [0]*len(chain_identifiers[key][1])]

    for pair in interactions:
        try:
            chain_id = pair['res_id'][0] # chain identifier is first character of the residue id
            interaction_id = '.'.join([identifier, chain_id]) # key for new dictionary
            index = chain_identifiers[chain_id][0].index(pair['res_id']) # index in sequence of binding residue

            interaction_list[interaction_id][1][index] = 1

        except KeyError: # some DNA-interacting chains are booted out for being too short which is shown by the KeyError
            continue

    # convert array of integers denoting annotations into a string
    for key in interaction_list:
        interaction_list[key][1] = [str(x) for x in interaction_list[key][1]]
        interaction_list[key][1] = ''.join(interaction_list[key][1])

    return interaction_list


def initial_binding_annotations(db, output):
    # used to track line in database
    prodb_line = 0

    for structure in iterate_json(db):
        residues = structure['protein']['residues']
        chains = structure['protein']['chains']
        nonstandard_residues = False
        residue_identifiers = dict() # dict of the form {residue_id: residue_name}
        chain_identifiers = dict() # dict of chains with form {'chain_id': ([res_ids], 'chain_sequence')}

        for residue in residues:
            if residue['chemical_modification'] is not None:
                nonstandard_residues = True
            residue_identifiers[residue['id']] = residue['name_short']

        # skip structure if chemically modified residues are detected
        if nonstandard_residues:
            continue

        for chain in chains:
            if chain['dna_interaction'] and len(chain['sequence']) >= 30:
                chain_identifiers[chain['chain_id']] = (chain['res_ids'], chain['sequence'])

        if len(chain_identifiers) == 0:
            continue

        interaction_list = get_contacts(chain_identifiers, structure)

        for key in interaction_list:
            pfile('>{}'.format(key), output)
            pfile('{}\n{}'.format(interaction_list[key][0], interaction_list[key][1]), output)


if __name__ == '__main__':
    args = parse_arguments()

    dnaprodb = [line for line in open_maybe_gzipped(args.infile)]
    initial_binding_annotations(dnaprodb, args.outfile)
