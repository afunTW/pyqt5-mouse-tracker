import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from filterpy.kalman import KalmanFilter

from src.utils import log_handler


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', dest='outdir', default='outputs')
    parser.add_argument('--nolog', dest='islog', action='store_false')
    parser.add_argument('--winname', dest='winname', default='kalman_filter mouse tracker')
    parser.set_defaults(islog=True)
    return parser

def main(args: argparse.Namespace):
    # setting log
    out_dir_path = Path(args.outdir)
    log_name = f"{datetime.now().strftime('%m%dT%H%M%S')}-{datetime.now().microsecond}.log"
    log_path = out_dir_path / log_name
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True)
    logger = logging.getLogger(__name__)
    log_handler(logger, logname=log_path if args.islog else None)
    logger.info(args)

if __name__ == "__main__":
    main(argparser().parse_args())
