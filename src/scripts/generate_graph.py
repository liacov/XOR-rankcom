"""
    It generates synthetic samples of a network having an intrinsic community and hierarchical
    structure.
    A collection of difference distributions can be used, varying the ratio parameter. More
    than one sample can be drawn from the same distribution.
"""

import sys

sys.path.append('../modules')

import os
import yaml
import numpy as np
from itertools import product
from argparse import ArgumentParser

import XOR_generator as gmXOR


def main(args):

    # Save relevant quantities in local variables
    settings = args.pop('settings') # remove settings from dict to avoid errors
    with open(settings) as f:
        conf = yaml.load(f, Loader = yaml.FullLoader)
    label = conf['label']
    folder = conf['folder']
    # Adjust ratio's range parameters
    args['max_ratio'] += args['step_ratio']

    # Generate ratio range
    ranges = [np.arange(*[*args.values()][3 * i: 3 * (i + 1)]) for i in range(len(args) // 3 - 1) ]
    if ranges == []:
        ranges = [np.array([args['min_ratio']])]

    # Generation loop
    for params in product(*ranges):
        params = list(map(lambda x: round(float(x.item()), 2), params))  # to have a clean yaml file
        conf['mu'] = params[0] # current ratio
        conf['label'] = label + '_r{}_XOR'.format(conf['mu']) # outfile label

        prng = np.random.RandomState(seed = 42)  # set seed random number generator

        if not os.path.exists(folder):
            os.makedirs(folder)

        for sn in range(args['samples']):
            conf['prng'] += prng.randint(500)
            # Save used setting to yaml file
            if conf['output_parameters']:
                fname = folder + 'setting' + '_' + conf['label'] + '_' + str(conf['prng'])
                with open(fname + '.yaml', 'w') as f:
                    yaml.dump(conf, f)
            # Generative class init
            print('\n--- Generating with XOR model ---')
            gen = gmXOR.SyntNetXOR(**conf)
            # Generate data
            _ = gen.EitherOr_planted_network()


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-ar', '--min_ratio', type = float, default = 0.0,
                   help = 'Element a of the interval [a,b) in which the ratio parameter is varied. Dafault = 0.0')
    p.add_argument('-br', '--max_ratio', type = float, default = 1.0,
                   help = 'Element b of the interval [a,b) in which the ratio parameter is varied. Default = 1.0')
    p.add_argument('-rstep', '--step_ratio', type = float, default = 0.1,
                   help = 'Step for exploring the interval [a,b) in which the ratio parameter is varied. Default = 0.1')
    p.add_argument('-n', '--samples', type = int, default = 1,
                   help = 'Number of synthetic samples to be drawn. Default = 1')
    p.add_argument('-s', '--settings', type = str, default = '../setting_syn_XOR.yaml',
                   help = 'Path for the setting file to use (yaml). Default = "../setting_syn_XOR.yaml"')
    args = p.parse_args()

    main(vars(args))
