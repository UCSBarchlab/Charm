import argparse
import logging

def get_parser():
    parser = argparse.ArgumentParser()
    return parser

def parse_args(parser):
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)
    return args

def addCommonOptions(parser):
    parser.add_argument('--log', action='store', dest='loglevel',
            default='info', help='Set log level to: info, debug, verbose.')

def addCompilerOptions(parser):
    parser.add_argument('source', action='store',
            help='Path to source .d file.')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='Verbose compiler.')
    parser.add_argument('--draw', action='store_true', default=False,
            help='Draw intermediate graphs out.')
    parser.add_argument('--z3core', action='store_true', default=False,
            help='Use Z3 as core engine.')
    parser.add_argument('--mcsamples', type=int, action='store', default=100,
            help='Number of samples to use for uncertain variables.')
