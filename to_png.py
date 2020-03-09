import os
import sys
import argparse
from pathlib import Path

import cv2
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from natsort import natsorted

from custom.utils import windowing_thorax
from custom.utils import windowing_brain
from custom.utils import write_png_image


def run(tfrecord_dir, png_dir):

    for exam_id in tqdm(natsorted(os.listdir(tfrecord_dir))):

    	for patient in natsorted(os.listdir(os.path.join(tfrecord_dir, exam_id))):
            files = sorted(glob(os.path.join(tfrecord_dir, exam_id, '**/*.dcm'), recursive=True))
	
            for file in files:
                try:
                    img_png = file.replace(tfrecord_dir, png_dir)
                    img_png = img_png.replace('.dcm', '.png')
                    folder = os.path.split(img_png)[0]
                    #print("file: {}\n folder: {}\n img_png: {}".format(
                    #    file, folder, img_png))
                    if not os.path.exists(folder):
                        Path(folder).mkdir(parents=True, exist_ok=True)

                    if os.path.exists(img_png):
                        continue
                    dcm = sitk.GetArrayFromImage(sitk.ReadImage(file))
                    npy = windowing_thorax(img_png, dcm)
                    write_png_image(img_png, npy)
                except:
                    print(file)
                    pass

#----------------------------------------------------------------------------
#_examples = 
'''

python3 to_png.py --tfrecord-dir /root/mini_nas/01.PGGAN/chestct/chestct_8bit_normal_lung_dcm --png-dir /root/projects/02.StyleGAN2/chestct/chestct_8bit_normal_rgb_png

'''


def main():
    parser = argparse.ArgumentParser(
	description='DICOM TO PNG'
        #epilog=_examples,
	#formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--tfrecord-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--png-dir', help='Dataset root directory', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.png_dir):
        print ('Error: Root directory for PNG format images does not exist.')
        sys.exit(1)

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()


