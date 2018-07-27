#!/usr/bin/env python
from __future__ import print_function
from keras.preprocessing import sequence
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPool1D
import collections as col
import pprint as pp # for debugging
import numpy as np
import argparse
import json
import gzip
import re
import os
import sys


np.set_printoptions(threshold=np.nan)

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


MAX_LEN = 1024


def perr(m):
    # print to stderr
    print(m, file=sys.stderr)


def open_maybe_gzipped(f):
    # open gzipped files
    with open(f, 'rb') as test_read:
        byte1, byte2 = test_read.read(1), test_read.read(2)
        if byte1 and ord(byte1) == 0x1f and byte2 and ord(byte2) == 0x8b:
            f = gzip.open(f, mode='rt')
        else:
            f = open(f, 'rt')
        return f


def parse_arguments():
    # parse_arguments
    parser = argparse.ArgumentParser(description='Specify input file')
    parser.add_argument('file', type=str, help='Collection file')
    return parser.parse_args()


def iterate_json(db):
    # generator for list of parsed json entries
    for line in db:
        entry = json.loads(line)
        yield entry


def DNA_sequences(ob):
    # extract strand sequences and corresponding nucleotide ID, return as tuple of type (OrderedDict, OrderedDict)

    # obtain sequence 1
    s1 = (ob['dna']['sequence_features']['sequence1'])

    s1_ids = col.OrderedDict()
    s2_ids = col.OrderedDict()

    # sequences in the nucleotides section are in order with strand1 nucleotides preceding strand2
    nuc_list = ob['dna']['nucleotides']

    # some name_short values are have extra characters due to chemical modification; only include if A, T, C, or G
    for i in range(0, len(s1)):
        s1_ids[nuc_list[i]['id']] = (''.join(re.findall('[ATCG]', nuc_list[i]['name'])), i)
    for i in range(len(s1), 2*len(s1)):
        s2_ids[nuc_list[i]['id']] = (''.join(re.findall('[ATCG]', nuc_list[i]['name'])), i-len(s1))

    return s1_ids, s2_ids


def protein_sequences(ob):
    # return OrderedDict of protein residue ID, protein name, and sequence number
    residues = ob['protein']['residues']
    s = col.OrderedDict()

    # sequences in the nucleotides section are in order with strand1 nucleotides preceding strand2
    for i in range(0, len(residues)):
        s[residues[i]['id']] = (residues[i]['name'], i)

    # dict of chains and residue identifiers that interact with DNA
    interacting_chains = dict()
    for chain in ob['protein']['chains']:
        if chain['dna_interaction']:
            interacting_chains[chain['chain_id']] = (chain['res_ids'], chain['sequence'])

    # s is of form  {res_id: (res_name, index)}, interacting_chains is of form {'chain_id': ([res_ids], sequence)}
    return s, interacting_chains


def contact_map(seq1, seq2, pro, ob, chain):
    # we want a 3D tensor contact map of the form p x d x 20 in which we neglect DNA sequence
    # we treat proteins in different chains in the same cocrystal structure as unique proteins
    # seq1, seq2, and pro are ordered dicts with IDs and chemical identity; chain is the dictionary with chain ID
    # residues in the chain, and amino acid sequence. Each tensor for the same co crystal structure will have
    # at most 2*number_of_chains, since each chain will have a tensor for both forward and reverse

    interactions = ob['interactions']['nucleotide-residue_interactions']

    # interaction_map will track contacts for each chain
    interaction_map1 = {}
    interaction_map2 = {}

    # create interaction map for each protein chain; should only be 1
    for key in chain.keys():
        interaction_map1[key] = np.zeros(shape=(len(chain[key][0]), len(seq1)))
        interaction_map2[key] = np.zeros(shape=(len(chain[key][0]), len(seq2)))

    # create interaction maps for each chain
    for it in interactions:
        # check if major or minor groove BASA value > 0 or if there are major/minor groove vdW or hbond interactions
        basa = it['basa']['sg']['total'] > 0 or it['basa']['wg']['total'] > 0
        hbond = it['hbond_sum']['wg']['total'] > 0 or it['hbond_sum']['sg']['total'] > 0
        vdw = it['vdw_sum']['wg']['total'] > 0 or it['vdw_sum']['wg']['total'] > 0

        if basa or hbond or vdw:
            try:
                # wrap in try / catch block. Some DNA-protein interactions are backbone. The DNA-protein interactions
                # dict is filtered for only chains that interact with DNA. KeyError means interaction is backbone only
                res_id = it['res_id']
                nuc_id = it['nuc_id']
                chain_id = res_id[0]
                protein_index = chain[chain_id][0].index(res_id)
                nuc_index = 0
                strand = 0

                if nuc_id in seq1.keys():
                    nuc_index = seq1[nuc_id][1]
                    strand = 1
                else:
                    nuc_index = seq2[nuc_id][1]
                    strand = 2

                if strand == 1:
                    interaction_map1[chain_id][protein_index][nuc_index] = 1
                if strand == 2:
                    interaction_map2[chain_id][protein_index][nuc_index] = 1

            except KeyError:
                continue

    return interaction_map1, interaction_map2


def mkdir(dir, mode=0o0750):
    """Construct a directory hierarchy using the given permissions."""
    if not os.path.exists(dir):
        os.makedirs(dir, mode)


def chain_names(ob):
    # get dictionary mapping chain ID to uniprot names
    the_names = {}
    for chain in ob['protein']['chains']:
        the_names[chain['chain_id']] = chain['uniprot_accession']
    return the_names


def rename_res(res):
    # convert 3 letter identifiers to 1 letter identifiers
    for key in res:
        try:
            # change dict
            res[key] = (PROTEIN_MAPPINGS[res[key][0]], res[key][1])
        # returns None if
        except KeyError:
            return None
    return res


def generate_pd_matrices(db, fwd, rev):
    # used to track line in database
    prodb_line = 0
    for ob in iterate_json(db):
        seq = DNA_sequences(ob) # tuple of OrderedDict with form {NucID: NucName} in order (fwd, rev_complement)
        res, interacting_chains = protein_sequences(ob) # OrderedDict of protein sequence and residue IDs
        # name = chain_names(ob) # commented out; originally used to make uniprot identifiers filenames

        # convert 3 letter identifiers to 1 letter identifiers; if chemically modified skip
        res = rename_res(res)
        if res is None:
            prodb_line += 1
            continue

        # ensure that exactly 1 interacting chain is present
        if len(interacting_chains.keys()) != 1:
            prodb_line += 1
            continue

        # generate p x d contact maps
        cmap1, cmap2 = contact_map(seq[0], seq[1], res, ob, interacting_chains)

        # dump the np arrays into directory of files. Fwd / rev in same file, can be accessed using ['fwd'] or ['rev']
        for key in cmap1:
            fwd['{}.fwd'.format(prodb_line)] = cmap1[key]
            rev['{}.rev'.format(prodb_line)] = cmap2[key]

        prodb_line += 1
    return fwd, rev


def generate_training_set(structures, fwd, rev):
    training_set_x = []
    training_set_y = []

    for key in fwd:
        # extract forward and reverse contact maps
        rev_key = key[:-3] + 'rev'
        cmap1 = fwd[key]
        cmap2 = rev[rev_key]

        # convert p x d matrix into sequence labels
        seq_label_1 = cmap1.sum(axis=1)
        seq_label_2 = cmap2.sum(axis=1)

        # skip if sequence label exceeds max length for optimization purposes
        if seq_label_1.shape[0] > MAX_LEN:
            continue

        assert seq_label_1.shape == seq_label_2.shape
        seq_label = np.zeros(seq_label_1.shape)

        # track both sequence labels
        for i in range(0, seq_label_1.shape[0]):
            if seq_label_1[i] == 1 or seq_label_2[i] == 1:
                seq_label[i] = 1

        # sometimes proteins will contact multiple nucleotides; truncate these to 1
        for x in np.nditer(seq_label, op_flags=['readwrite']):
            x[...] = 1 if x[...] >= 1 else 0
        training_set_y.append(seq_label)

        # get the json object, extract the 1 x p x 20 sequence label
        entry_num = int(re.match('[0-9]+', key).group(0))
        ob = json.loads(structures[entry_num])
        train = np.zeros((seq_label.shape[0], 20))

        # generate the input one-hot encoded protein sequence
        res, interacting_chains = protein_sequences(ob)  # OrderedDict of protein sequence and residue IDs
        res = rename_res(res)

        # generate the p x 20 input matrix
        for id in res:
            channel = AMINO_ALPHABET.index(res[id][0])
            position = res[id][1]
            train[position][channel] = 1
        training_set_x.append(train)

    for arr in training_set_y:
        arr = arr.resize((MAX_LEN, ), refcheck=False)

    for arr in training_set_x:
        arr = arr.resize((MAX_LEN, 20), refcheck=False)

    # should be  2D tensor of type (batch_size, max)
    training_set_x = np.stack(training_set_x)

    # should be 3D tensor of type (batch, max, 20)
    training_set_y = np.stack(training_set_y)

    print('Input of shape: {}'.format(training_set_x.shape))
    print('Output of shape: {}'.format(training_set_y.shape))
    return training_set_x, training_set_y


def create_model(train_x, train_y, validate_x, validate_y):
    # set parameters
    batch_size = 32
    output_dim = MAX_LEN
    epochs = 1
    kernel_size = (16)
    filters = 128
    activation = 'relu'
    padding = 'same'

    perr('Building model...')
    model = Sequential()
    model.add(Conv1D(filters,
                     kernel_size,
                     padding=padding,
                     activation=activation,
                     strides=1))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding=padding,
                     activation=activation,
                     strides=1))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding=padding,
                     activation=activation,
                     strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(validate_x, validate_y))


def main():
    args = parse_arguments()
    perr('Loading db...')
    db = [line for line in open_maybe_gzipped(args.file)]
    perr('Done...')

    # store contact maps
    perr('Generating p x d matrices....')
    fwd = dict()
    rev = dict()
    fwd, rev = generate_pd_matrices(db, fwd, rev)
    perr('Done...')

    # go through and generate tensors containing sequence label from the contact maps
    perr('Generating training sets...')
    train_x, train_y = generate_training_set(db, fwd, rev)
    perr('Done...')

    # Divide into training and validation set. Hardcoded to save last 50 structures as validation set TODO:
    validate_x = train_x[-50:, :, :]
    validate_y = train_y[-50:, :]
    train_x = train_x[:-50, :, :]
    train_y = train_y[:-50, :]

    # create and run keras model
    # perr('Creating and training model')
    # create_model(train_x, train_y, validate_x, validate_y)

    # check percentage of zeros in validation and training sets
    validate_pct = np.count_nonzero(validate_y)/validate_y.shape[0]/validate_y.shape[1]
    train_pct = np.count_nonzero(train_y)/train_y.shape[0]/train_y.shape[1]
    perr('Nonzero in validation y: {}, Nonzero in train y: {}'.format(validate_pct, train_pct))


if __name__ == '__main__':
    main()
