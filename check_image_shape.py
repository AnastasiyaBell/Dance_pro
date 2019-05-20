import os
import argparse

from scipy import misc


parser = argparse.ArgumentParser()
parser.add_argument(
    "file_names",
    help="Paths to files which shapes are checked."
)
parser.add_argument(
    "--shape",
    "-s",
    help="Required shape. If shape of an image does not match, "
         "file name is printed.",
    nargs='+',
    type=int,
)
parser.add_argument(
    "--dir",
    "-d",
    help="If provided, file names is interpreted as a directory "
         "which content needs to be checked.",
    action="store_true",
)
args = parser.parse_args()

if args.dir:
    file_names = map(
        lambda x: os.path.join(args.file_names, x),
        os.listdir(os.path.expanduser(args.file_names)),
    )
    file_names = filter(
        os.path.isfile,
        file_names,
    )
else:
    file_names = args.file_names

for path in file_names:
    image = misc.imread(path)
    sh = image.shape
    if list(sh) != args.shape:
        print(path, sh, end='\n')
