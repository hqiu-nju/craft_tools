import numpy as np
from sigpyproc.Readers import FilReader
import matplotlib.pyplot as plt

#### direct dedispersion file creator

__author__ ="Hao Qiu"
def _main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='Script description', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument("-d","--dm",type=float,default=0,help="set dm value for dedispersion")
    parser.add_argument("-f","--file",type=str,default="frb_dedispersed.npy",help="set outputfile")
    parser.add_argument("-f","--file",type=str,default="frb_dedispersed.npy",help="set outputfile")
    parser.add_argument(dest='files', nargs='+')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    _main()
