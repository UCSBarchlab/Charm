import argparse
import logging
from Charm.models import MathModel
from Charm.models import RiskFunctionCollection

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
    parser.add_argument('--use-energy', action='store_true',
            default=False, help='Use energy computation.')

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

def addModelOptions(parser):
    parser.add_argument('--math-model', action='store', dest='math_model',
            default='symmetric', choices=MathModel.names(),
            help='Select math model to use.')
    parser.add_argument('--risk-func', action='store', dest='risk_func',
            default='linear', choices=RiskFunctionCollection.funcs,
            help='Select risk model to use.')
    parser.add_argument('--trans', action='store_true', dest='trans', default=False,
            help='Use transformed Gaussian as inputs.')
    parser.add_argument('--use-power', action='store_true', dest='use_power',
            default=False, help='use power equations.')

def addApplicationOptions(parser):
    parser.add_argument('--f', action='store', type=float, default=.9,
            help='Set f value.')
    parser.add_argument('--c', action='store', type=float, default=.01,
            help='Set c value.')

def addRegressionOptions(parser):
    parser.add_argument('--fc-file', action='store', dest='fc_file', default=None,
            help='filepath to empirical workload for fc regression.')
    parser.add_argument('--cpudb-dir', action='store', dest='cpudb_dir', default=None,
            help='path to cpudb directory.')

def addIOOptions(parser):
    parser.add_argument('--save-path', action='store', default=None,
            help='File path to dump result.')
    parser.add_argument('--load-path', action='store', default=None,
            help='File path to load result.')
