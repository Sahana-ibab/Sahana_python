#!/usr/bin/env python
from optparse import OptionParser
import shutil
import glob
import os
import subprocess
import sys

################################################################################
# install_data.py
#
# Download and arrange pre-trained models and data.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    parser.add_option('-r', dest='restart', default=False, action='store_true',
                      help='Do not overwrite existing files, as if restarting an aborted installation [Default: %default]')
    (options, args) = parser.parse_args()

    # Check for wget or curl
    if shutil.which('wget'):
        dl_cmd = 'wget'
    elif shutil.which('curl'):
        dl_cmd = 'curl -L -O'
    else:
        print('Cannot find wget or curl to download files', file=sys.stderr)
        exit(1)

    os.chdir('data')

    ############################################################
    # download pre-trained model
    ############################################################
    os.chdir('models')

    if not options.restart or not os.path.isfile('pretrained_model.th'):
        print('Downloading pre-trained model.', file=sys.stderr)

        cmd = f'{dl_cmd} https://www.dropbox.com/s/rguytuztemctkf8/pretrained_model.th.gz'
        subprocess.call(cmd, shell=True)

        cmd = 'gunzip pretrained_model.th.gz'
        subprocess.call(cmd, shell=True)

    os.chdir('..')

    ############################################################
    # download human genome
    ############################################################
    os.chdir('genomes')

    if not options.restart or not os.path.isfile('hg19.fa'):
        print('Downloading hg19 FASTA from UCSC. If you already have it, CTL-C and place a symlink in the genomes directory named hg19.fa',
              file=sys.stderr)

        # download hg19
        cmd = f'{dl_cmd} https://hgdownload.cse.ucsc.edu/goldenPath/hg19/bigZips/chromFa.tar.gz -O chromFa.tar.gz'
        subprocess.call(cmd, shell=True)

        # un-tar
        cmd = 'tar -xzvf chromFa.tar.gz'
        subprocess.call(cmd, shell=True)

        # cat
        cmd = 'cat chr?.fa chr??.fa > hg19.fa'
        subprocess.call(cmd, shell=True)

        # clean up
        os.remove('chromFa.tar.gz')
        for chrom_fa in glob.glob('chr*.fa'):
            os.remove(chrom_fa)

    if not options.restart or not os.path.isfile('hg19.fa.fai'):
        cmd = 'samtools faidx hg19.fa'
        subprocess.call(cmd, shell=True)

    os.chdir('..')

    ############################################################
    # download and prepare public data
    ############################################################
    if not options.restart or not os.path.isfile('encode_roadmap.h5'):
        cmd = f'{dl_cmd} https://www.dropbox.com/s/h1cqokbr8vjj5wc/encode_roadmap.bed.gz'
        subprocess.call(cmd, shell=True)
        cmd = 'gunzip encode_roadmap.bed.gz'
        subprocess.call(cmd, shell=True)

        cmd = f'{dl_cmd} https://www.dropbox.com/s/8g3kc0ai9ir5d15/encode_roadmap_act.txt.gz'
        subprocess.call(cmd, shell=True)
        cmd = 'gunzip encode_roadmap_act.txt.gz'
        subprocess.call(cmd, shell=True)

        # make a FASTA file
        cmd = 'bedtools getfasta -fi genomes/hg19.fa -bed encode_roadmap.bed -s -fo encode_roadmap.fa'
        subprocess.call(cmd, shell=True)

        # make an HDF5 file
        cmd = 'seq_hdf5.py -c -t 71886 -v 70000 encode_roadmap.fa encode_roadmap_act.txt encode_roadmap.h5'
        subprocess.call(cmd, shell=True)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
