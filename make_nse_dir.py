
"""Relabel a collection of images, introducing an NSE ("no sound event") label
given a threshold and a file containing a score for all images.
"""

import os
import sys

from os.path import exists

import click

from adak.sed import SoundEventDetectionFilter


@click.command(help=__doc__)
@click.argument('source')
@click.argument('target')
@click.argument('nse_file')
@click.option('-t', '--nse-threshold', default=0.1, show_default=True)
def main(source, target, nse_file, nse_threshold):
    if exists(target):
        sys.exit('E: target exists.')

    os.system(f'cp -r {source} {target}')

    sed = SoundEventDetectionFilter(nse_file, threshold=nse_threshold)
    sed.relabel_files(target)


if __name__ == '__main__':
    main()
