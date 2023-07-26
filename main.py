"""
Written by: Peter Lionel Harry Newman, 2023, (p.newman @ sydney edu au)
Helpful for students
1. GUI interface to:
2. search a folder for .czi files
3. export mips of various kinds for each czi channel
4. morphometrics about the images
Go check out the Allen Institue of Cell Science package !!!
This exists to help students with mip generation, and because of the bugs in the mosaic builder
"""

# std
import csv
from functools import partial
import itertools
from multiprocessing import Pool, cpu_count
import pickle
import platform
import re
import time
import warnings
import xml.etree.ElementTree as ET

# 3rd
import aicspylibczi
import attr
from cellpose import models
import cv2
from itertools import product
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageChops, ImageTk
import pyvista as pv
from readlif.reader import LifFile
from scipy.ndimage import zoom, binary_fill_holes, distance_transform_edt, binary_dilation, binary_erosion
from skimage.measure import regionprops, label
from skimage import morphology
from tkinter.filedialog import askdirectory
import tkinter as tk
from tkinter import ttk
from torch import device
from tqdm import tqdm
import webcolors


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

global SO

@attr.s(auto_attribs=True, auto_detect=True)
class ProcessOptions:
    # image path
    image_path: str = attr.ib(default='')
    save_path: str = attr.ib(default='')

    # save options
    save_mip_channels: bool = attr.ib(default=False)
    save_mip_panel: bool = attr.ib(default=True)
    save_mip_merge: bool = attr.ib(default=False)
    save_dye_overlaid: bool = attr.ib(default=True)
    save_colors: bool = attr.ib(default=True)
    use_multiprocessing: bool = attr.ib(default=False)

    # segmentation options
    segment_image: bool = attr.ib(default=False)
    mask_ch: int = attr.ib(default=0)
    cyto: bool = attr.ib(default=False)
    nuc: bool = attr.ib(default=False)
    channels_of_interest: list = attr.ib(default=[0])
    # same_for_all_images: bool = attr.ib(default=False)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'image_path':
            self.save_path = os.path.join(os.path.dirname(value), 'save')


@attr.s(auto_attribs=True, auto_detect=True)
class SegmentationOptions:

    mask_channel_var: int = attr.ib(default=0)
    cyto_var: bool = attr.ib(default=False)
    nuc_var: bool = attr.ib(default=True)
    segment_2D: bool = attr.ib(default=True)
    segment_3D: bool = attr.ib(default=False)
    same_for_all_images_var: bool = attr.ib(default=True)
    channel_vars: list = attr.ib(default=[0])
    mask_channel: int = attr.ib(default=0)
    channels_of_interest_vars: list = attr.ib(default=[])


@attr.s(auto_attribs=True, auto_detect=True)
class MaskProperties2D:

    centroid: np.array = attr.ib(default=np.array([[],[]]).T)
    id: np.array = attr.ib(default=np.array([]))
    area: np.array = attr.ib(default=np.array([]))
    perimeter: np.array = attr.ib(default=np.array([]))
    form_factor: np.array = attr.ib(default=np.array([]))
    minor_ax: np.array = attr.ib(default=np.array([]))
    major_ax: np.array = attr.ib(default=np.array([]))
    eccentricity: np.array = attr.ib(default=np.array([]))
    convexity: np.array = attr.ib(default=np.array([]))
    orientation: np.array = attr.ib(default=np.array([]))
    mask_sat: np.array = attr.ib(default=np.array([]))
    tf_sat: np.array = attr.ib(default=np.array([]))
    dist_from_edge: np.array = attr.ib(default=np.array([]))


@attr.s(auto_attribs=True, auto_detect=True)
class MaskProperties3D:

    id: np.array = attr.ib(default=np.array([]))
    volume: np.array = attr.ib(default=np.array([]))
    centroid: np.array = attr.ib(default=np.array([[], [], []]).T)
    surface_area: np.array = attr.ib(default=np.array([]))
    sphericity: np.array = attr.ib(default=np.array([]))
    major_ax: np.array = attr.ib(default=np.array([]))
    minor_ax: np.array = attr.ib(default=np.array([]))
    least_ax: np.array = attr.ib(default=np.array([]))
    eccentricity: np.array = attr.ib(default=np.array([]))
    convexity: np.array = attr.ib(default=np.array([]))
    orientation: np.array = attr.ib(default=np.array([]))
    mask_sat: np.array = attr.ib(default=np.array([]))
    tf_sat: np.array = attr.ib(default=np.array([[], []]).T)


@attr.s(auto_attribs=True, auto_detect=True)
class BioImage:
    """
    czi image class for lite image processing
    wrapper class around the ACIS wrapper class;
    around the czilib library
    """

    path: str = attr.ib(default='')
    imfile: aicspylibczi.CziFile = attr.ib(default=None)
    czi: tuple = attr.ib(default=())
    im: np.ndarray = attr.ib(default=[0., 0.])
    cell_mass: np.ndarray = attr.ib(default=[0., 0.])
    mosaic: np.ndarray = attr.ib(default=[0., 0.])
    num_channels: int = attr.ib(default=0)
    num_z_slices: int = attr.ib(default=0)
    height: int = attr.ib(default=0)
    width: int = attr.ib(default=0)
    num_timepoints: int = attr.ib(default=0)
    num_scenes: int = attr.ib(default=0)
    num_blocks: int = attr.ib(default=0)
    num_mosaics: int = attr.ib(default=0)
    metadata: str = attr.ib(default='')
    nmip: np.ndarray = attr.ib(default=[0., 0.])
    mip: np.ndarray = attr.ib(default=[0., 0.])
    colours: list = attr.ib(default=[])
    dyes: list = attr.ib(default=[])
    scale: np.array = attr.ib(default=[1., 1., 1.])
    bbox: np.array = attr.ib(default=np.array([0., 0.]))

    mask: np.ndarray = attr.ib(default=np.array([]))
    mask_ch: int = attr.ib(default=0)
    mask_props: MaskProperties2D = attr.ib(default=MaskProperties2D())

    big_image: bool = attr.ib(default=False)

    def load_tif(self):

        """
        load a tif file
        """

        # get the number of channels, z_slices and shape
        os.chdir(os.path.dirname(self.path))

        all_images = []
        self.num_channels = 0

        # search for tif files with same name
        for root, _, files in os.walk(os.path.dirname(self.path)):
            for file in files:
                if file.endswith('.tif'):
                    if file[0] == '.':
                        continue
                    if os.path.basename(self.path)[:-4] in file\
                            and not file == os.path.basename(self.path):
                        all_images.append(file)
                        self.num_channels += 1

        self.num_z_slices = 1

        if not all_images:
            print('No tif files found')
            return

        (self.width, self.height, _) = plt.imread(all_images[0]).shape
        self.metadata = ''
        self.scale = -1

        self.im = np.zeros((self.num_channels, self.num_z_slices, self.height, self.width))

        self.colours = []
        self.dyes = []

        # convert im to f64
        for c in range(self.num_channels):
            im = plt.imread(all_images[c]).astype(np.float64)

            # Get the R, G, and B channels
            r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
            # r, g, b = im[:, :, 1], im[:, :, 2], im[:, :, 0]

            # Calculate the grayscale image
            self.im[c, 0, :, :] = 0.2989 * r + 0.5870 * g + 0.1140 * b

            color = [r.max(), g.max(), b.max()]
            color /= np.max(color)
            color = (color * 255).astype(int)
            self.colours.append(color)
            self.dyes.append(webcolors.rgb_to_name(color))

        self.scale = np.array([1., 1., 1.])

    def load_czi(self):
        """
        load a czi file
        """

        # get the image
        if self.path.endswith('.czi'):
            self.imfile = aicspylibczi.CziFile(self.path)
        else:
            # Set color to red
            print('\033[91m', end='')
            print(f'Failed to process: {self.path}')
            # Reset color to default
            print('\033[0m', end='')

        dims = self.imfile.get_dims_shape()[0]

        if dims['X'][1] == 0 and dims['Y'][1] == 0:
            return 'metadata_only'

        # get the number of channels, z_slices and shape
        self.num_channels = dims['C'][1]
        self.num_z_slices = dims['Z'][1]
        self.width = dims['X'][1]
        self.height = dims['Y'][1]

        self.metadata = self.imfile.meta
        # looks like the first 3 distance/values are camera props?
        self.scale = np.array([float(self.metadata.findall('.//Distance/Value')[i].text) for i in range(3, 6)])[::-1]
        self.scale = self.scale * 1e6

        # load each image
        if self.imfile.is_mosaic():

            # additional checks
            if not 'M' in dims:
                raise ValueError('Mosaic image found, but no M dimension found')

            # load in all the bounding boxes
            self.num_mosaics = dims['M'][1]
            self.bbox = np.zeros((self.num_mosaics, 4)).astype(int)
            for m in range(self.num_mosaics):
                _ = self.imfile.get_mosaic_tile_bounding_box(C=0, Z=0, M=m)
                self.bbox[m, :] = int(_.x), int(_.y), int(_.w), int(_.h)

            # check that w and h is the same
            if not np.all(self.bbox[:, 2] == self.bbox[0, 2]) or not np.all(self.bbox[:, 3] == self.bbox[0, 3]):
                raise ValueError('Mosaic bounding boxes are not the same size')

            # simplify the box
            self.bbox[:, 0] = self.bbox[:, 0] - np.min(self.bbox[:, 0])
            self.bbox[:, 1] = self.bbox[:, 1] - np.min(self.bbox[:, 1])

            # initialize the image
            self.width = np.max(self.bbox[:, 0]) + self.bbox[0, 2]
            self.height = np.max(self.bbox[:, 1]) + self.bbox[0, 3]

            # get the 'im' == (czyx)
            self.mosaic = np.moveaxis(self.imfile.read_image()[0],
                                    [self.imfile.dims.index('C'),
                                     self.imfile.dims.index('Z'),
                                     self.imfile.dims.index('Y'),
                                     self.imfile.dims.index('X'),
                                     self.imfile.dims.index('M')],
                                    [0, 1, 2, 3, 4])

            # index the last dimensions at 0
            for i in range(len(dims) - 5):
                self.mosaic = self.mosaic[..., 0]

            # look at the size of im and cry if really large
            if (self.mosaic.nbytes / 1e9) > 5:
                self.big_image = True
                print(f'WARNING: Image size is {self.mosaic.nbytes / 1e9} GB, this may cause memory issues')
                return

            # move the mosaics into im - some rounding bug? means I need to add 1 to the width and height
            self.im = np.zeros((self.num_channels, self.num_z_slices, self.height + 1, self.width + 1))

            print(f'loading mosaic')
            for m in range(self.num_mosaics):
                if m % (self.num_mosaics // 10) == 0:
                    print(f'.', end='')
                self.im[:, :, self.bbox[m, 1]:self.bbox[m, 1] + self.bbox[m, 3], self.bbox[m, 0]:self.bbox[m, 0] + self.bbox[m, 2]] = self.mosaic[:, :, :, :, m]

        else:
            self.im = self.imfile.read_image()

            # get the 'im' == (czyx)
            self.im = np.moveaxis(self.im[0],
                                  [self.imfile.dims.index('C'),
                                   self.imfile.dims.index('Z'),
                                   self.imfile.dims.index('Y'),
                                   self.imfile.dims.index('X')],
                                  [0, 1, 2, 3])

            # index the last dimensions at 0
            for i in range(len(dims) - 4):
                self.im = self.im[..., 0]

        # convert im to f64
        self.im = self.im.astype(np.float64)

        # get the other info:
        if 'T' in dims: # timepoints
            # self.num_timepoints = dims['T'][1]
            # warnings.warn('this script throws away this info')
            pass
        if 'S' in dims: # scenes
            pass
        if 'B' in dims: # blocks
            pass
        if 'V' in dims:
            # The V-dimension ('view').
            pass
        if 'I' in dims:
            # The I-dimension ('illumination').
            pass
        if 'R' in dims:
            # The R-dimension ('rotation').
            pass
        if 'H' in dims:
            # The H-dimension ('phase').
            pass

    def load_lif(self):
        """
        load a lif file
        """

        # get the image
        if path.endswith('.lif'):
            self.imfile = LifFile(self.path)
        else:
            # Set color to red
            print('\033[91m', end='')
            print(f'Failed to process: {path}')
            # Reset color to default
            print('\033[0m', end='')
            return

        dims = self.imfile.image_list[0]

        if dims['dims'].x == 0 and dims['dims'].y == 0:
            return 'metadata_only'

        # get the number of channels, z_slices and shape
        self.num_channels = dims['channels']
        self.num_z_slices = dims['dims'].z
        self.width = dims['dims'].x
        self.height = dims['dims'].y

        # get scale
        self.scale = np.array(self.imfile.image_list[0]['scale'][0:3])
        if self.scale[2] == None:
            self.scale[2] = 0
        self.scale = self.scale[::-1] # xyz to zyx

        # load each image for mosaics
        if dims['dims'].m > 1:

            # Set color to red
            print('\033[91m', end='')
            print(f'Pete hasn''t implemened mosaics yet:')
            # Reset color to default
            print('\033[0m', end='')
            return

            # TBD additional checks
            if not 'M' in dims:
                raise ValueError('Mosaic image found, but no M dimension found')

            # TDB look at the size of im and cry if really large
            if (self.mosaic.nbytes / 1e9) > 5:
                self.big_image = True
                print(
                    f'WARNING: Image size is {self.im.nbytes / 1e9} GB, this will cause memory issues')
                return

        # load each image for non-mosaics
        else:
            # create an empty array C-Z-Y-X
            self.im = np.zeros((self.num_channels, self.num_z_slices, self.height, self.width))
            for c in range(self.num_channels):
                for z in range(self.num_z_slices):
                    self.im[c, z, :, :] = self.imfile.get_image(0).get_frame(z=z, t=0, c=c)

        # convert im to f64
        self.im = self.im.astype(np.float64)

        # get the other info:
        if dims['dims'].t > 1: # timepoints
            token = 'T'
            self.num_timepoints = dims['dims'].t
            warnings.warn(f'this script throws away this info: {token}')
            pass

    def extract_colors(self):

        if self.path.endswith('.lif'):
            self.metadata = xml_to_dict(self.imfile.xml_header)

            # not sure whats up with lif files, they dont seem to store laser or dye info?
            results = []
            def find_key(dictionary, target):
                for key, value in dictionary.items():
                    if key == target:
                        results.append(value)
                    elif isinstance(value, dict):
                        find_key(value, target)
            find_key(self.metadata, 'dye')
            find_key(self.metadata, 'color')
            find_key(self.metadata, 'laser')

            # assuming colors are in order blue to red
            self.colours = []
            self.dyes = []

            for c in range(self.num_channels):
                if c == 0:
                    self.colours.append([0, 255, 255])
                    self.dyes.append('cyan')
                elif c == 1:
                    self.colours.append([255, 255, 0])
                    self.dyes.append('yellow')
                elif c == 2:
                    self.colours.append([255, 0, 0])
                    self.dyes.append('red')
                elif c == 3:
                    self.colours.append([255, 0, 255])
                    self.dyes.append('magenta')
                elif c == 4:
                    self.colours.append([255, 255, 255])
                    self.dyes.append('white')

        elif self.path.endswith('.czi'):
            self.metadata = self.imfile.meta
            dyes = self.metadata.findall('.//DyeName')

            # looks like the first 3 distance/values are camera props?
            self.scale = np.array([float(self.metadata.findall('.//Distance/Value')[i].text) for i in range(3, 6)])[::-1]

            if len(dyes) != self.num_channels:
                warnings.warn('num channel != num dyes')
                return

            self.dyes = [None] * self.num_channels

            for c in range(self.num_channels):
                self.dyes[c] = dyes[c].text

                if 'DAPI' in self.dyes[c]\
                        or 'dapi' in self.dyes[c] \
                        or 'Hoechst 33342' in self.dyes[c] \
                        or 'Hoechst 33258' in self.dyes[c]:
                    self.colours.append([0, 255, 255])
                    continue
                elif 'FITC' in self.dyes[c]:
                    self.colours.append([255, 255, 0])
                    continue
                elif 'Cy3' in self.dyes[c] or 'Rhodamine' in self.dyes[c]:
                    self.colours.append([255, 0, 0])
                    continue
                elif 'Cy5' in self.dyes[c]:
                    self.colours.append([255, 0, 255])
                    continue

                # extract all numbers from dye
                try:
                    dye_nums = float(re.findall(r'\d+', self.dyes[c])[0])
                except:
                    dye_nums = -1

                if dye_nums < 0:
                    self.colours.append([255, 255, 255])
                elif dye_nums < 405:
                    self.colours.append([0, 255, 255])
                elif dye_nums < 500:
                    self.colours.append([255, 255, 0])
                elif dye_nums < 600:
                    self.colours.append([255, 0, 0])
                elif dye_nums < 700:
                    self.colours.append([255, 0, 255])
                else:
                    self.colours.append([0, 0, 0])
                    warnings.warn(f'no color found for {self.path}; channel {c}, {self.dyes[c]}')

    def project_mip(self, side_projections=False, z_scale=1):
        """
        make a maximum intensity projection
        """
        # check for z slices
        if self.num_z_slices == 1:

            print('\033[96m', end='')
            print(f'no z slices found in {self.path} returning')
            print('\033[0m', end='')
            self.mip = np.zeros((self.num_channels,
                                 self.im.shape[2],
                                 self.im.shape[3]))
            for c in range(self.num_channels):
                self.mip[c, :, :] = self.im[c, 0, :, :]
            return

        # initialize the mip
        if side_projections:

            if not self.big_image:
                self.mip = np.zeros((self.num_channels,
                                     self.im.shape[2] + self.im.shape[1] * z_scale + 1,
                                     self.im.shape[3] + self.im.shape[1] * z_scale + 1))

                for c in range(self.num_channels):
                    # check for z slices
                    self.mip[c,
                            0:self.im.shape[2],
                            0:self.im.shape[3]] = np.max(self.im[c, :, :, :], axis=0)

                    projection_yz = np.max(self.im[c, :, :, :], axis=1)
                    projection_yz = zoom(projection_yz, (z_scale, 1))
                    self.mip[c,
                            (self.im.shape[2] + 1):(self.im.shape[2] + 1 + self.im.shape[1] * z_scale),
                            0:self.im.shape[3]] = projection_yz

                    projection_xz = np.max(self.im[c, :, :, :], axis=2).T
                    projection_xz = zoom(projection_xz, (1, z_scale))
                    self.mip[c,
                            0:self.im.shape[2],
                            (self.im.shape[3] + 1):(self.im.shape[3] + 1 + self.im.shape[1]  * z_scale)] = projection_xz
                return

            if self.big_image:

                self.mip = np.zeros((self.num_channels,
                                     self.height + self.num_z_slices * z_scale + 1,
                                     self.width + self.num_z_slices * z_scale + 1))

                for m in tqdm(range(self.num_mosaics), desc=f'Generating mip for *big image*:'):
                    if m % (self.num_mosaics // 10) == 0:
                        print(f'.', end='')

                    for c in range(self.num_channels):
                        self.mip[c,
                                self.bbox[m, 1]:self.bbox[m, 1] + self.bbox[m, 3],
                                self.bbox[m, 0]:self.bbox[m, 0] + self.bbox[m, 2]] = np.max(self.mosaic[c, :, :, :, m], axis=0)


                        projection_yz = np.max(self.mosaic[c, :, :, :, m], axis=1)
                        projection_yz = zoom(projection_yz, (z_scale, 1))
                        self.mip[c,
                                -(self.num_z_slices * z_scale + 1):-1,
                                self.bbox[m, 0]:self.bbox[m, 0] + self.bbox[m, 2]] = projection_yz

                        projection_xz = np.max(self.mosaic[c, :, :, :, m], axis=2).T
                        projection_xz = zoom(projection_xz, (1, z_scale))
                        self.mip[c,
                                self.bbox[m, 1]:self.bbox[m, 1] + self.bbox[m, 3],
                                -(self.num_z_slices * z_scale + 1):-1,] = projection_xz
                return

        else:
            if not self.big_image:
                self.mip = np.zeros((self.num_channels,
                                     self.im.shape[2],
                                     self.im.shape[3]))

                for c in range(self.num_channels):
                    self.mip[c, :, :] = np.max(self.im[c, :, :, :], axis=0)

                return

            if self.big_image:
                self.mip = np.zeros((self.num_channels,
                                     self.height + 1,
                                     self.width + 1))

                for m in tqdm(range(self.num_mosaics), desc=f'Generating mip for *big image*:'):
                    for c in range(self.num_channels):
                        self.mip[c,
                        self.bbox[m, 1]:self.bbox[m, 1] + self.bbox[m, 3],
                        self.bbox[m, 0]:self.bbox[m, 0] + self.bbox[m, 2]] = np.max(self.mosaic[c, :, :, :, m], axis=0)



    def normalize(self, max=False, gamma=1):
        """
        modifies '.mip' normalizing the image
        """
        if self.mip == [0, 0]:
            print('No mip to normalize')
            return

        self.nmip = np.zeros((self.num_channels, self.mip.shape[2], self.mip.shape[1]))

        for c in range(self.num_channels):

            if max:
                mip = (self.mip[c, :, :] - np.nanmin(self.mip[c, :, :])) / \
                                      (np.nanmax(self.mip[c, :, :]) - np.nanmin(self.mip[c, :, :]))
            else:
                mip = self.mip[c, :, :] - np.nanmin(self.mip[c, :, :])
                if np.nansum(mip) != 0:
                    mip_p = np.percentile(mip, 99.8)
                    mip[mip > mip_p] = mip_p
                    mip = mip/mip_p

            if gamma != 1:
                mip = mip ** gamma

            self.nmip[c, :, :] = mip * 255

    def save(self,
             save_path,
             save_mip_channels,
             save_mip_panel,
             save_mip_merge,
             save_dye_overlaid,
             save_colors):

        if self.mip == [0, 0]:
            print('No mip to save')
            return

        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(self.path))
        except:
            os.mkdir(os.path.dirname(self.path))
            os.chdir(os.path.dirname(self.path))

        if self.big_image:
            optimize = False
        else:
            optimize = True

        file_stem = os.path.splitext(os.path.basename(self.path))[0]

        for c in range(self.num_channels):
            # PIL on each channel
            mip = Image.fromarray(self.mip[c, :, :])
            mip = mip.convert('L')
            base = np.ceil(self.num_channels ** 0.5).astype('int')

            if save_colors and len(self.colours[c]) > 0:
                mip = ImageOps.colorize(mip, (0, 0, 0), tuple(self.colours[c]))
            else:
                mip = mip.convert('RGB')

            if save_dye_overlaid:
                font_color = tuple(self.colours[c])
                font_size = self.height // 50
                draw = ImageDraw.Draw(mip)
                if platform == 'linux' or platform == 'linux2' or platform == 'darwin':
                    text_overlay = [[]]
                    text_overlay.append(c * ['\n'])
                    text_overlay.append([self.dyes[c]])
                    text_overlay = ''.join([item for sublist in text_overlay for item in sublist])
                    draw.text((0, 0), text_overlay,
                              font_color,
                              ImageFont.truetype('Arial.ttf', size=font_size))
                elif platform == 'win32':
                    draw.text((0, 0), self.dyes[c],
                              font_color,
                              ImageFont.truetype('arial.ttf', size=font_size))

            # create an image of all 'c' merged
            if save_mip_merge or \
                    save_mip_panel and self.num_channels > 1 and self.num_channels < base ** 2:
                if c == 0:
                    self.mip_merge = mip.copy()
                else:
                    self.mip_merge = Image.merge('RGB', (
                        ImageChops.add(self.mip_merge.getchannel('R'), mip.getchannel('R')),
                        ImageChops.add(self.mip_merge.getchannel('G'), mip.getchannel('G')),
                        ImageChops.add(self.mip_merge.getchannel('B'), mip.getchannel('B'))))


            if c == 0:
                self.mip_panel = np.zeros((mip.height * base, mip.width * base, 3))

            self.mip_panel[c % base * mip.height:(c % base + 1) * mip.height,
                      c // base * mip.width:(c // base + 1) * mip.width,
                      :] = np.array(mip).copy()

            # save
            if save_mip_channels:
                # mip.save(f'{file_stem}_ch{c}_.png', optimize=optimize)
                cv2.imwrite(f'{file_stem}_ch{c}_.png', cv2.cvtColor(np.array(mip), cv2.COLOR_RGB2BGR))

        # save merged images
        if save_mip_merge:
            # self.mip_merge.save(f'{file_stem}_merge.png', optimize=optimize)
            cv2.imwrite(f'{file_stem}_merge.png', cv2.cvtColor(np.array(self.mip_merge), cv2.COLOR_RGB2BGR))

        # save panel images
        if save_mip_panel and self.num_channels > 1:
            # add the merge to the panel if there is space
            if self.num_channels < base ** 2:
                self.mip_panel[self.num_channels % base * mip.height:(self.num_channels % base + 1) * mip.height,
                               self.num_channels // base * mip.width:(self.num_channels // base + 1) * mip.width,
                               :] \
                                    = np.array(self.mip_merge).copy()

            # remove black space
            self.mip_panel = self.mip_panel[~np.all(self.mip_panel == 0, axis=(1, 2))]

            # save mip panel cv2 (is much faster)
            cv2.imwrite(f'{file_stem}_panel.png',
                        cv2.cvtColor(self.mip_panel.astype('uint8'), cv2.COLOR_RGB2BGR))

        print(f'converted {file_stem} and saved')
        os.chdir(cwd)

    def combine_big_image_masks(self, mask):

        mask = np.transpose(mask, (2, 1, 0))

        # define mosaic bounds
        window_size = 200
        overlap = 100

        mask = mask[::-1]

        yl = np.hstack([0, np.arange(window_size - overlap, mask.shape[1], window_size)])
        yu = np.hstack([window_size, np.arange(window_size - overlap + window_size, mask.shape[1], window_size), mask.shape[1]])

        xl = np.hstack([0, np.arange(window_size - overlap, mask.shape[2], window_size)])
        xu = np.hstack([window_size, np.arange(window_size - overlap + window_size, mask.shape[2], window_size), mask.shape[2]])

        # process a mosaic of the areas (speeds up processing significantly due to morphological ops)
        my = np.vstack([yl, yu]).T
        mx = np.vstack([xl, xu]).T

        mosaic = product(my, mx)

        if not np.any(mask):
            print(f'\033[94mError: {self.path} has no labels\033[0m')
            return

        combine_threshold = 0.7

        for nm, m in enumerate(mosaic):

            # get the mask for this mosaic
            vol = mask[:, m[1][0]:m[1][1], m[0][0]:m[0][1]]

            for nz in range(1, vol.shape[0]):

                slice_j = vol[nz, :, :]
                slice_i = vol[nz - 1, :, :]

                # # get id for slices
                # slice_j_labels = np.unique(slice_j)
                # slice_i_labels = np.unique(slice_i)

                check_labels_mask = slice_i - slice_j < 0
                # check_labels_j = slice_j_labels * check_labels_mask
                # check_labels_i = slice_i_labels * check_labels_mask
                check_labels_j = slice_j * check_labels_mask
                check_labels_i = slice_i * check_labels_mask

                uni_slice_j = np.unique(check_labels_j)
                uni_slice_i = np.unique(check_labels_i)
                uni_slice_j = uni_slice_j[uni_slice_j != 0]
                uni_slice_i = uni_slice_i[uni_slice_i != 0]

                # check that theres something in both slices
                if reduction(uni_slice_j, np.logical_or, 'any', None, None, None) \
                        and reduction(uni_slice_i, np.logical_or, 'any', None, None, None):

                    for j in uni_slice_j:
                        for i in uni_slice_i:
                            img_j = np.array(slice_j == j)
                            img_i = np.array(slice_i == i)
                            img_ij = np.logical_and(img_i, img_j)
                            sum_img_j = img_j.sum()
                            if sum_img_j > 2000:  # remove the object if its got a really big 2D size note doesnt work for slice 0 objects
                                slice_j[slice_j == j] = 0
                                changed = True
                            elif reduction(img_ij, np.logical_or, 'any', None, None, None):
                                # if an object on adjacent layers has an intersection > 'combine_threshold' make them the same
                                sum_img_i = img_i.sum()
                                sum_img_ij = img_ij.sum()
                                if sum_img_ij / sum_img_i > combine_threshold or sum_img_ij / sum_img_j > combine_threshold:
                                    slice_j[slice_j == j] = i
                                    changed = True

                    if changed:
                        mask[nz, m[1][0]:m[1][1], m[0][0]:m[0][1]] = slice_j

        # condense the mask
        mask = np.max(mask, axis=0)

        # convert back to xy
        mask = np.transpose(mask, (1, 0))

        return mask

    def process_2d_mask(self):

        print('processing 2D mask...')

        # process vars
        count = 0
        try:
            SO.channels_of_interest_vars[0].get()
            ch_of_interest = [i.get() for i in SO.channels_of_interest_vars]
        except:
            ch_of_interest = [i for i in SO.channels_of_interest_vars]

        self.mask_props = MaskProperties2D()

        if not len(self.mask.shape) == 2:
            print(f'\033[94mError: {self.path} isn''t 2D?\033[0m')
            return

        # rescale image to make yx of an pixel the same, then 1 µm, or if scale < 1 µm, scale.min()
        min_scale = np.min(np.hstack([self.scale[1::], 1]))
        zoom_by = self.scale[1::] / min_scale

        im = np.zeros(np.hstack([self.num_channels, (np.array(self.mip.shape[1::]) * zoom_by).astype('int')]))
        for c in range(self.num_channels):
            im[c, :, :] = zoom(np.squeeze(self.mip[c, :, :]), zoom_by, order=1)

        mask = zoom(self.mask, zoom_by, order=0)

        # define mosaic bounds
        window_size = 200
        overlap = 100

        yl = np.hstack([0, np.arange(window_size - overlap, im.shape[1], window_size)])
        yu = np.hstack([window_size, np.arange(window_size - overlap + window_size, im.shape[1], window_size), im.shape[1]])


        xl = np.hstack([0, np.arange(window_size - overlap, im.shape[2], window_size)])
        xu = np.hstack([window_size, np.arange(window_size - overlap + window_size, im.shape[2], window_size), im.shape[2]])

        # process a mosaic of the areas (speeds up processing significantly due to morphological ops)
        my = np.vstack([yl, yu]).T
        mx = np.vstack([xl, xu]).T

        mosaic = product(my, mx)

        if not np.any(mask):
            print(f'\033[94mError: {self.path} has no labels\033[0m')
            return

        mask_id_processed = np.array([0])

        for nm, m in enumerate(mosaic):

            # get the mask for this mosaic
            area = mask[m[1][0]:m[1][1], m[0][0]:m[0][1]]
            area_im = im[:, m[1][0]:m[1][1], m[0][0]:m[0][1]]

            # if there are no labels in this area, skip
            if not np.any(area):
                continue

            # remove mask ids that have already been processed
            area_id = np.unique(area)

            # if area_id is in mask_id_processed, remove it
            for i in np.intersect1d(area_id, mask_id_processed):
                area[area == i] = 0

            area_id = np.unique(area)
            area_id = area_id[area_id != 0]

            rp = regionprops(area)

            for rn, r in enumerate(rp):
                # filter out large & small objects
                if r.area < 10 or r.area > 10_000:
                    continue

                # save properties
                self.mask_props.id = np.hstack((self.mask_props.id, area_id[rn]))
                self.mask_props.area = np.hstack((self.mask_props.area, r.area))
                self.mask_props.centroid = np.vstack((self.mask_props.centroid, np.array(r.centroid) + np.array([m[1][0], m[0][0]])))
                self.mask_props.perimeter = np.hstack((self.mask_props.perimeter, r.perimeter))
                self.mask_props.form_factor = np.hstack((self.mask_props.form_factor, r.perimeter ** 2 / r.area))
                self.mask_props.minor_ax = np.hstack((self.mask_props.minor_ax, r.minor_axis_length))
                self.mask_props.major_ax = np.hstack((self.mask_props.major_ax, r.major_axis_length))
                self.mask_props.eccentricity = np.hstack((self.mask_props.eccentricity, r.eccentricity))
                self.mask_props.convexity = np.hstack((self.mask_props.convexity, r.convex_area / r.area))
                self.mask_props.orientation = np.hstack((self.mask_props.orientation, r.orientation))

                # saturation intensity in mask of the other channels
                tf_sat = np.hstack([
                    np.mean(area_im[c][r.coords[:, 0], r.coords[:, 1]])
                    for c in range(self.num_channels) if ch_of_interest[c]
                ])

                if count == 0:
                    self.mask_props.tf_sat = tf_sat
                else:
                    self.mask_props.tf_sat = np.vstack((self.mask_props.tf_sat, tf_sat))

                count += 1

            # add all area_id to mask_id_processed
            mask_id_processed = np.hstack([area_id, mask_id_processed])

        # calculate distance of each cell in mask from edge of cell mass
        print('Calculating distances of cells from edge of cell mass...')


        # cellpose_model = models.Cellpose(gpu=True, model_type='nuclei')
        # nuc_mask = cellpose_model.eval(
        #     self.nmip[nuclear_channel].astype(np.uint8),
        #     diameter=diam,
        #     flow_threshold=flow_threshold,
        #     cellprob_threshold=cellprob_threshold)[0]
        # # region props centroids
        # properties = regionprops(label(nuc_mask))
        # centroids = np.array([prop.centroid for prop in properties]).astype(np.int64)
        #
        # threshold = 1
        # cell_mass = (nuc_mask > 0 * (threshold + 1)).astype(np.float64)
        # tri_upper_ind = np.triu_indices(centroids.shape[0], k=1)
        # centroid_combinations = centroids[np.array(tri_upper_ind)]
        #
        # # thin out centroid combinations
        # dist = centroid_combinations[0, :, :] - centroid_combinations[1, :, :]
        # dist = (dist[:, 0] ** 2 + dist[:, 1] ** 2) ** 0.5
        # dist = dist < cell_mass.shape[1] / 2
        # centroid_combinations = centroid_combinations[:, dist, :]
        #
        # z = np.zeros_like(cell_mass).astype(np.float64)
        # for c1, c2 in tqdm(centroid_combinations.transpose((1, 0, 2)), total=centroid_combinations.shape[1]):
        #     temp = z.copy()
        #     cell_mass += cv2.line(temp, c1[::-1], c2[::-1], (1,), 5).astype(np.float64) / centroid_combinations.shape[1]
        #

        combo = self.nmip.astype(np.float64)
        combo = combo.sum(axis=0)
        combo = combo / combo.max() * 255
        combo = combo.astype(np.uint8)
        # bw = 255 - cv2.adaptiveThreshold(combo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)

        combo = cv2.medianBlur(combo, 33)
        th, _ = cv2.threshold(combo, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw = combo > (th * 0.33)
        bw = np.pad(bw, 2, 'constant', constant_values=255)
        bw[0:2,0:2] = 0
        bw[0:2,-3:-1] = 0
        bw[-3:-1,0:2] = 0
        bw[-3:-1,-3:-1] = 0
        bw = bw > 0
        bw = binary_fill_holes(bw)
        bw = bw[2:-2, 2:-2]
        diam = np.mean(self.mask_props.area) + 3 * np.std(self.mask_props.area)
        bw = morphology.remove_small_objects(bw, int(3 * (diam/2)**2))

        # calculate distance of each mask from the edge of the cell mass
        self.mask_props.dist_from_edge = np.zeros(self.mask_props.id.shape[0])
        dist_transform = distance_transform_edt(bw)
        for n, i in enumerate(self.mask_props.id):
            obj = (mask == i)
            if obj.sum() == 0:
                continue
            mean_distance = np.nanmean(dist_transform[obj])
            if np.isfinite(mean_distance):
                self.mask_props.dist_from_edge[n] = mean_distance
            else:
                self.mask_props.dist_from_edge[n] = -1

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(im.transpose((1, 2, 0)))
        ax[1].imshow(bw * 255)
        cell_mass_im = im[0, :, :] / 2 + bw.astype(np.uint8) * 255 / 2
        ax[2].imshow(cell_mass_im)
        ax[2].scatter(self.mask_props.centroid[:, 1],
                      self.mask_props.centroid[:, 0],
                      c='r',
                      s=self.mask_props.dist_from_edge / 50)

        self.cell_mass = cell_mass_im

    def process_3d_mask(self):

        print('processing 3D mask...')

        # process vars
        count = 0
        try:
            SO.channels_of_interest_vars[0].get()
            ch_of_interest = [i.get() for i in SO.channels_of_interest_vars]
        except:
            ch_of_interest = [i for i in SO.channels_of_interest_vars]

        self.mask_props = MaskProperties3D()

        if not len(self.mask.shape) == 3:
            print(f'\033[94mError: {self.path} isn''t 3D?\033[0m')
            return

        # rescale image to make xyz of a voxel the same, then 1 µm, or if scale < 1 µm, scale.min()
        min_scale = np.min(np.hstack([self.scale[1::], 1]))
        zoom_by = self.scale / min_scale

        im = np.zeros(np.hstack([self.num_channels, np.round((np.array(self.mask.shape) * zoom_by)).astype('int')]))
        for c in range(self.num_channels):
            im[c, :, :, :] = zoom(self.im[c, :, :, :], zoom_by, order=1)

        mask = zoom(self.mask, zoom_by, order=0)

        # define mosaic bounds
        window_size = 200
        overlap = 100

        zl = np.hstack([0, np.arange(window_size - overlap, im.shape[0], window_size)])
        zu = np.hstack([window_size, np.arange(window_size - overlap + window_size, im.shape[0], window_size), im.shape[0]])

        yl = np.hstack([0, np.arange(window_size - overlap, im.shape[1], window_size)])
        yu = np.hstack([window_size, np.arange(window_size - overlap + window_size, im.shape[1], window_size), im.shape[1]])

        xl = np.hstack([0, np.arange(window_size - overlap, im.shape[2], window_size)])
        xu = np.hstack([window_size, np.arange(window_size - overlap + window_size, im.shape[2], window_size), im.shape[2]])

        # process a mosaic of the areas (speeds up processing significantly due to morphological ops)
        mz = np.vstack([zl, zu]).T
        my = np.vstack([yl, yu]).T
        mx = np.vstack([xl, xu]).T

        mosaic = product(mz, my, mx)

        if not np.any(mask):
            print(f'\033[94mError: {self.path} has no labels\033[0m')
            return

        mask_id_processed = np.array([0])

        combine_threshold = 0.7

        for nm, m in enumerate(mosaic):

            # get the mask for this mosaic
            vol = mask[:, m[1][0]:m[1][1], m[0][0]:m[0][1]]

            for nz in range(1, vol.shape[0]):

                slice_j = vol[nz, :, :]
                slice_i = vol[nz - 1, :, :]

                # get id for slices
                slice_j_labels = np.unique(slice_j)
                slice_i_labels = np.unique(slice_i)

                check_labels_mask = slice_i - slice_j < 0
                check_labels_j = slice_j_labels * check_labels_mask
                check_labels_i = slice_i_labels * check_labels_mask

                uni_slice_j = np.unique(check_labels_j)
                uni_slice_i = np.unique(check_labels_i)
                uni_slice_j = uni_slice_j[uni_slice_j != 0]
                uni_slice_i = uni_slice_i[uni_slice_i != 0]

                # check that theres something in both slices
                if reduction(uni_slice_j, np.logical_or, 'any', None, None, None) \
                        and reduction(uni_slice_i, np.logical_or, 'any', None, None, None):

                    for j in uni_slice_j:
                        for i in uni_slice_i:
                            img_j = np.array(slice_j == j)
                            img_i = np.array(slice_i == i)
                            img_ij = np.logical_and(img_i, img_j)
                            sum_img_j = img_j.sum()
                            if sum_img_j > 2000:  # remove the object if its got a really big 2D size note doesnt work for slice 0 objects
                                slice_j[slice_j == j] = 0
                                changed = True
                            elif reduction(img_ij, np.logical_or, 'any', None, None, None):
                                # if an object on adjacent layers has an intersection > 'combine_threshold' make them the same
                                sum_img_i = img_i.sum()
                                sum_img_ij = img_ij.sum()
                                if sum_img_ij / sum_img_i > combine_threshold or sum_img_ij / sum_img_j > combine_threshold:
                                    slice_j[slice_j == j] = i
                                    changed = True

                    if changed:
                        mask[nz, m[1][0]:m[1][1], m[0][0]:m[0][1]] = slice_j

        for nm, m in enumerate(mosaic):

            # get the mask for this mosaic
            vol = mask[m[2][0]:m[2][1], m[1][0]:m[1][1], m[0][0]:m[0][1]]
            vol_im = im[:, m[2][0]:m[2][1], m[1][0]:m[1][1], m[0][0]:m[0][1]]

            # if there are no labels in this area, skip
            if not np.any(vol):
                continue

            # remove mask ids that have already been processed
            area_id = np.unique(vol)

            # if area_id is in mask_id_processed, remove it
            for i in np.intersect1d(area_id, mask_id_processed):
                vol[vol == i] = 0

            area_id = np.unique(vol)
            area_id = area_id[area_id != 0]

            rp = regionprops(vol)

            for rn, r in enumerate(rp):
                # filter out large & small objects
                if r.area < 10 or r.area > 10_000:
                    continue

                # save properties
                self.mask_props.id = np.hstack((self.mask_props.id, area_id[rn]))
                self.mask_props.volume = np.hstack((self.mask_props.volume, r.area))
                self.mask_props.centroid = np.vstack((self.mask_props.centroid, np.array(r.centroid) + np.array([m[2][0], m[1][0], m[0][0]])))
                self.mask_props.bounding_box = np.vstack((self.mask_props.bounding_box, r.bbox))
                self.mask_props.form_factor = np.hstack((self.mask_props.form_factor, r.perimeter ** 2 / r.area))
                self.mask_props.eccentricity = np.hstack((self.mask_props.eccentricity, r.eccentricity))
                self.mask_props.orientation = np.hstack((self.mask_props.orientation, r.orientation))

                # saturation intensity in mask of the other channels
                tf_sat = np.hstack([
                    np.mean(vol_im[c][r.coords[:, 0], r.coords[:, 1], r.coords[:, 2]])
                    for c in range(self.num_channels) if ch_of_interest[c]
                ])

                if count == 0:
                    self.mask_props.tf_sat = tf_sat
                else:
                    self.mask_props.tf_sat = np.vstack((self.mask_props.tf_sat, tf_sat))

                count += 1

            # add all area_id to mask_id_processed
            mask_id_processed = np.hstack([area_id, mask_id_processed])

    def plot_2d_mask(self):

        try:
            SO.channels_of_interest_vars[0].get()
            ch_of_interest = [i.get() for i in SO.channels_of_interest_vars]
        except:
            ch_of_interest = [i for i in SO.channels_of_interest_vars]

        # with num_channel axes
        fig, ax = plt.subplots(2,
                               np.max([sum(ch_of_interest) + 1, 3]),
                               figsize=(4 * np.max([3, sum(ch_of_interest) + 1]), 8))

        for a in ax.flatten():
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor('black')
        fig.set_facecolor('black')

        x = self.mask_props.centroid[:,1]
        y = self.mask_props.centroid[:,0]

        colors = ['cyan', 'yellow', 'red', 'magenta', 'green']
        for c in range(self.mask_props.tf_sat.shape[1]):
            ax[0, c+1].scatter(x, y,
                          s=(self.mask_props.tf_sat[:, c] / self.mask_props.tf_sat[:, c].max()),
                          c=colors[c],
                          alpha=0.99)

        # ax[1, 0].imshow(self.mip[self.mask_ch, :, :], cmap='hot')
        # ax[1, 0].invert_yaxis()
        # ax[1, 1].imshow(self.mask, cmap='hot')
        # ax[1, 1].invert_yaxis()

        # plt.tight_layout()
        # plt.show()

    def plot_3d_mask(self):

        try:
            SO.channels_of_interest_vars[0].get()
            ch_of_interest = [i.get() for i in SO.channels_of_interest_vars]
        except:
            ch_of_interest = [i for i in SO.channels_of_interest_vars]

        # Create PyVista plotter
        plotter = pv.Plotter(shape=(1, sum(ch_of_interest) + 1),
                             window_size=(300 * (sum(ch_of_interest) + 1), 300))
        plotter.set_background('black')

        # Get centroids and properties
        x = self.mask_props.centroid[:, 0]
        y = self.mask_props.centroid[:, 1]
        z = self.mask_props.centroid[:, 2]
        centroids = np.column_stack((x, y, z))

        # Create glyph source
        glyph_source = pv.PolyData(centroids)

        # Add mask saturation glyphs
        if self.mask_props.mask_sat.size > 0:
            mask_sat_scale = (self.mask_props.mask_sat / self.mask_props.mask_sat.max())
        else:
            mask_sat_scale = np.array([])

        plotter.subplot(0, 0)
        plotter.add_glyph(glyph_source, scale=mask_sat_scale, color='white', render_points_as_spheres=True)
        plotter.add_title("Mask Saturation", color='white')

        # Add glyphs for other channels
        colors = ['cyan', 'yellow', 'red', 'magenta', 'green']
        for c in range(self.mask_props.tf_sat.shape[1]):
            if self.mask_props.tf_sat[:, c].size > 0:
                tf_sat_scale = (self.mask_props.tf_sat[:, c] / self.mask_props.tf_sat[:, c].max())
            else:
                tf_sat_scale = np.array([])

            plotter.subplot(0, c + 1)
            plotter.add_glyph(glyph_source, scale=tf_sat_scale, color=colors[c], render_points_as_spheres=True)
            plotter.add_title(f"Ch of Int Sat {c}", color='white')

        # Show the plot
        plotter.show()

    def save_2d_mask(self, po):

        print(f'saving segmentation results{self.path}')

        if not os.path.exists(po.save_path):
            os.mkdir(po.save_path)

        os.chdir(po.save_path)
        file_stem = os.path.basename(self.path)

        if self.big_image:
            optimize = False
        else:
            optimize = True

        # save mask image
        mask = self.mask.astype('uint16')

        # gray to rgb
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3)).astype('uint8')

        # loop through mask for each object and make it a random color
        for c in range(3):
            mask_c = mask
            for i in range(1, mask.max()):
                mask_c[mask_c == i] = 30 + np.random.randint(0, 255 - 30)

            mask_color[:, :, c] = mask_c

        # save images
        cv2.imwrite(f'{file_stem}_masks.png', cv2.cvtColor(np.array(mask), cv2.COLOR_GRAY2BGR))
        cv2.imwrite(f'{file_stem}_masks_color.png', mask_color)
        cv2.imwrite(f'{file_stem}_cell_mass.png', self.cell_mass)

        # save data as csv
        with open(file_stem + '_2Dresults.csv', 'w') as f:
            # column names
            var = 'id,' \
                  'area,' \
                  'centroid_x,' \
                  'centroid_y,' \
                  'perimeter,' \
                  'form_factor,' \
                  'minor_ax,' \
                  'major_ax,' \
                  'eccentricity,' \
                  'convexity,' \
                  'orientation,' \
                  'dist_from_edge'
            tf = ''
            try:
                SO.channels_of_interest_vars[0].get()
                for i in range(sum([i.get() for i in SO.channels_of_interest_vars])):
                    tf += f',ch_of_int_sat{i}'
            except:
                for i in range(sum([i for i in SO.channels_of_interest_vars])):
                    tf += f',ch_of_int_sat{i}'

            var = var + tf + '\n'

            # write column names
            f.write(var)

            for i in range(self.mask_props.id.size):
                # write row
                tf = ''
                for c in range(self.mask_props.tf_sat.shape[1]):
                    tf += f',{self.mask_props.tf_sat[i, c]}'
                f.write(f'{self.mask_props.id[i]},'
                        f'{self.mask_props.area[i]},'
                        f'{self.mask_props.centroid[i, 0]},'
                        f'{self.mask_props.centroid[i, 1]},'
                        f'{self.mask_props.perimeter[i]},'
                        f'{self.mask_props.form_factor[i]},'
                        f'{self.mask_props.minor_ax[i]},'
                        f'{self.mask_props.major_ax[i]},'
                        f'{self.mask_props.eccentricity[i]},'
                        f'{self.mask_props.convexity[i]},'
                        f'{self.mask_props.orientation[i]},'
                        f'{self.mask_props.dist_from_edge[i]},'
                        + tf + '\n')

    def save_3d_mask(self, po):

        print(f'saving segmentation results{self.path}')

        if not os.path.exists(po.save_path):
            os.mkdir(po.save_path)

        os.chdir(po.save_path)
        file_stem = os.path.basename(self.path)

        if self.big_image:
            optimize = False
        else:
            optimize = True

        # save mask image
        mask = Image.fromarray(np.max(self.mask.astype('uint16'), axis=0))
        mask.save(f'{file_stem}_masks.png', optimize=optimize)

        # save data as csv
        with open(file_stem + '_3Dresults.csv', 'w') as f:
            # column names
            var = 'id,' \
                  'volume,' \
                  'centroid_x,' \
                  'centroid_y,' \
                  'centroid_z,' \
                  'surface_area,' \
                  'sphericity,' \
                  'minor_ax,' \
                  'major_ax,' \
                  'least_ax,' \
                  'eccentricity,' \
                  'convexity,' \
                  'orientation,' \
                  'mask_sat,' \
                  'tf_sat'

            tf = ''
            try:
                SO.channels_of_interest_vars[0].get()
                for i in range(sum([i.get() for i in SO.channels_of_interest_vars])):
                    tf += f',ch_of_int_sat{i}'
            except:
                for i in range(sum([i for i in SO.channels_of_interest_vars])):
                    tf += f',ch_of_int_sat{i}'


            var = var + tf + '\n'

            # write column names
            f.write(var)

            for i in range(self.mask_props.id.size):
                # write row
                tf = ''
                for c in range(self.mask_props.tf_sat.shape[1]):
                    tf += f',{self.mask_props.tf_sat[i, c]}'
                f.write(f'{self.mask_props.id[i]},'
                        f'{self.mask_props.volume[i]},'
                        f'{self.mask_props.centroid[i, 0]},'
                        f'{self.mask_props.centroid[i, 1]},'
                        f'{self.mask_props.centroid[i, 2]},'
                        f'{self.mask_props.surface_area[i]},'
                        f'{self.mask_props.sphericity[i]},'
                        f'{self.mask_props.minor_ax[i]},'
                        f'{self.mask_props.major_ax[i]},'
                        f'{self.mask_props.least_ax[i]},'
                        f'{self.mask_props.eccentricity[i]},'
                        f'{self.mask_props.convexity[i]},'
                        f'{self.mask_props.orientation[i]},'
                        f'{self.mask_props.mask_sat[i]}' + tf + '\n')


# @attr.s(auto_attribs=True, auto_detect=True)
class MainGUI(tk.Tk):
    """
    Create a tkinter window to select options:
    - select a folder to search for czi files
    - select check boxes for:
        - side projections (w/ side project scaling)
        - gamma correction
        - save mip channels
        - save mip panel
        - save mip merge
        - save dye overlaid
        - save colors
        - use multiprocessing
    - select a folder to save the images to
    """

    def __init__(self):
        # create a tkinter window
        self.root = tk.Tk()
        self.root.title(' ')

        # initalize variables
        self.search_path = tk.StringVar()
        # self.search_path.set('.set/search/path')
        self.search_path.set('/Users/peternewman/Desktop/im/ben')
        self.save_path = tk.StringVar()
        # self.save_path.set('.set/save/path')
        self.save_path.set('/Users/peternewman/Desktop/im/ben')

        # establish a scale factor
        s = 1.0

        w = int(420 * s)
        h = int(250 * s)

        # size the window
        self.root.minsize(w, h)
        self.root.maxsize(w, h)
        self.root.geometry(f'{w}x{h}')

        # add a title
        tk.Label(self.root, text='czi2png', font=('Arial', 25))\
            .place(relx=10 / w, rely=10 / h) # width=135/w, height=36/h)

        # # add search and save path buttons
        tk.Button(self.root, text='Search Path', command=self.specify_search_path,
                  width=8, height=1).place(relx=10 / w, rely=53 / h)
        tk.Button(self.root, text='Save Path', command=self.specify_save_path,
                  width=8, height=1).place(relx=10 / w, rely=83 / h)

        self.search_path_label = tk.Label(self.root, textvariable=self.search_path, font=('Arial', 12), fg='gray')\
            .place(relx=124 / w, rely=58 / h) # width=135/w, height=36/h)
        self.save_path_label = tk.Label(self.root, textvariable=self.save_path, font=('Arial', 12), fg='gray') \
            .place(relx=124 / w, rely=88 / h)  # width=135/w, height=36/h)

        # add check box for file type czi, lif or tif
        self.file_type = tk.StringVar()
        self.file_type.set('czi')
        tk.Radiobutton(self.root, text='czi', variable=self.file_type,
                       value='czi') \
            .place(relx=10 / w, rely=140 / h)
        tk.Radiobutton(self.root, text='lif', variable=self.file_type,
                       value='lif') \
            .place(relx=80 / w, rely=140 / h)
        tk.Radiobutton(self.root, text='tif', variable=self.file_type,
                       value='tif') \
            .place(relx=150 / w, rely=140 / h)

        # # add a button to run the program
        tk.Button(self.root, text='Convert 2 png!', command=self.main,
                  width=20, height=2).place(relx=10 / w, rely=180 / h)

        # # add checkbox to save mip channels, mip panel, mip merge, dye overlaid, colors, multiprocessing
        tk.Label(self.root, text='Options', font=('Arial', 12)) \
                        .place(relx=260 / w, rely=22 / h)  # width=135/w, height=36/h)

        self.save_mip_channels = tk.BooleanVar()
        self.save_mip_channels.set(False)
        tk.Checkbutton(self.root, text='save mip channels', command = self.display_input,
                       variable=self.save_mip_channels, onvalue=1, offvalue=0)\
                        .place(relx=260 / w, rely=53 / h)

        self.save_mip_panel = tk.BooleanVar()
        self.save_mip_panel.set(False)
        tk.Checkbutton(self.root, text='save mip panel', command = self.display_input,
                       variable=self.save_mip_panel, onvalue=1, offvalue=0) \
                        .place(relx=260 / w, rely=80 / h)

        self.save_mip_merge = tk.BooleanVar()
        self.save_mip_merge.set(True)
        tk.Checkbutton(self.root, text='save mip merge', command = self.display_input,
                       variable=self.save_mip_merge, onvalue=1, offvalue=0) \
                        .place(relx=260 / w, rely=107 / h)

        self.save_dye_overlaid = tk.BooleanVar()
        self.save_dye_overlaid.set(False)
        tk.Checkbutton(self.root, text='save dye overlaid', command = self.display_input,
                       variable=self.save_dye_overlaid, onvalue=1, offvalue=0) \
                        .place(relx=260 / w, rely=133 / h)

        self.save_colors = tk.BooleanVar()
        self.save_colors.set(True)
        tk.Checkbutton(self.root, text='save colors', command = self.display_input,
                       variable=self.save_colors, onvalue=1, offvalue=0) \
                        .place(relx=260 / w, rely=160 / h)

        self.use_multiprocessing = tk.BooleanVar()
        self.use_multiprocessing.set(False)
        tk.Checkbutton(self.root, text='use multiprocessing', command = self.display_input,
                       variable=self.use_multiprocessing, onvalue=1, offvalue=0) \
                        .place(relx=260 / w, rely=186 / h)

        self.segment_image = tk.BooleanVar()
        self.segment_image.set(True)
        tk.Checkbutton(self.root, text='segment image',
                       variable=self.segment_image, onvalue=1, offvalue=0) \
                        .place(relx=260 / w, rely=213 / h)

        self.root.mainloop()

    # debugging
    def display_input(self):
        print(f'search path: {self.search_path}')
        print(f'save path: {self.save_path}')
        print(f'save mip channels: {self.save_mip_channels.get()}')
        print(f'save mip panel: {self.save_mip_panel.get()}')
        print(f'save mip merge: {self.save_mip_merge.get()}')
        print(f'save dye overlaid: {self.save_dye_overlaid.get()}')
        print(f'save colors: {self.save_colors.get()}')
        print(f'use multiprocessing: {self.use_multiprocessing.get()}')

    def specify_search_path(self,):
        self.search_path.set(tk.filedialog.askdirectory(parent=self.root, initialdir='/',
                                        title='Please select a directory'))

    def specify_save_path(self,):
        self.save_path.set(tk.filedialog.askdirectory(parent=self.root, initialdir='/',
                                        title='Please select a directory'))

    def main(self):
        try:
            if self.search_path.get() == '.set/search/path' or self.save_path.get() == '.set/save/path':
                tk.messagebox.showerror('Python Error', 'please select Search and Save paths * unassigned *')
                return

            if not os.path.isdir(self.search_path.get()) or not os.path.isdir(self.save_path.get()):
                tk.messagebox.showerror('Python Error', 'Search and Save paths not directories')
                return

            # check that at least one save option is selected
            if self.save_mip_channels.get() + self.save_mip_panel.get() + self.save_mip_merge.get() < 1:
                tk.messagebox.showerror('Python Error', 'select at least channels, panel or merge image to save')
                return
        except:
            print('incomplete error checks')

        # set path of image files
        search_path = self.search_path.get()

        # get images
        image_files = find_all_images_in_path(search_path, filetype=self.file_type.get())
        all_process_files = [None] * len(image_files)

        for n, image_file in enumerate(image_files):
            po = ProcessOptions()
            po.save_path = self.save_path.get()
            po.save_mip_channels = self.save_mip_channels.get()
            po.save_mip_panel = self.save_mip_panel.get()
            po.save_mip_merge = self.save_mip_merge.get()
            po.save_dye_overlaid = self.save_dye_overlaid.get()
            po.save_colors = self.save_colors.get()
            po.use_multiprocessing = self.use_multiprocessing.get()
            po.segment_image = self.segment_image.get()
            all_process_files[n] = po
            all_process_files[n].image_path = image_file

        self.root.destroy()

        im_file_sizes = [os.path.getsize(image_file) for image_file in image_files]
        cumulative_file_size = np.cumsum(im_file_sizes)

        process_file(all_process_files[0])

        # pop all_process_files[0] off the list
        all_process_files.pop(0)

        # time the function
        start_time = time.perf_counter()
        # run the processing routine on the czi images
        if all_process_files[0].use_multiprocessing:
            print(f'running multi processed: {image_files}')
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                p.map(process_file, all_process_files)

        else:
            print(f'running single threaded on: {image_files}')
            for n, process_param in enumerate(all_process_files):
                process_file(process_param)

                # calculate time remaining use the files size to estimate processing time
                time_remaining = (time.perf_counter() - start_time) / \
                                 cumulative_file_size[n] * \
                                 (cumulative_file_size[-1] -
                                  cumulative_file_size[n])
                # time taken
                # per byte
                # * bytes remaining

                print(f'Time remaining: {time_remaining:0.2f} seconds')

        # print the run time
        print(
            f'\nTime elapsed: {time.perf_counter() - start_time:0.2f} seconds')

        # display a message box to indicate that the processing is complete
        tk.messagebox.showinfo('Python Info', 'Images saved as png')

        # close the window
        self.root.destroy()

@attr.s(auto_attribs=True, auto_detect=True)
class SegmentationGUI:
    def __init__(self, parent, image):
        self.root = parent
        self.root.title('Segmentation Options')

        # the reason we are here
        global SO
        SO = SegmentationOptions()

        window_size = (1500, 550)
        self.root.geometry(f"{window_size[0]}x{window_size[1]}")

        image.mip = np.array(image.mip)
        self.image = image

        self.mask_channel_var = tk.IntVar()
        self.mask_channel_var.set(0)
        SO.mask_channel_var = self.mask_channel_var.get()

        self.cyto_var = tk.BooleanVar()
        self.cyto_var.set(False)
        SO.cyto_var = self.cyto_var.get()

        self.nuc_var = tk.BooleanVar()
        self.nuc_var.set(True)
        SO.nuc_var = self.nuc_var.get()

        self.segment_2D = tk.BooleanVar()
        self.segment_2D.set(True)
        SO.segment_2D = self.segment_2D.get()

        self.segment_3D = tk.BooleanVar()
        self.segment_3D.set(False)
        SO.segment_3D = self.segment_3D.get()

        self.same_for_all_images_var = tk.BooleanVar()
        self.same_for_all_images_var.set(False)
        SO.same_for_all_images_var = self.same_for_all_images_var.get()

        tk.Label(self.root, text="Mask channel:").place(relx=0.01, rely=0.05)
        ttk.Combobox(self.root, values=list(range(self.image.num_channels)), textvariable=self.mask_channel_var, width=int(0.005 * window_size[0])).place(relx=0.1, rely=0.05)

        self.cyto_checkbox = tk.Checkbutton(self.root, text="cyto", variable=self.cyto_var, command=self.toggle_nuc)
        self.cyto_checkbox.place(relx=0.01, rely=0.1)

        self.nuc_checkbox = tk.Checkbutton(self.root, text="nuc", variable=self.nuc_var, command=self.toggle_cyto)
        self.nuc_checkbox.place(relx=0.1, rely=0.1)

        tk.Label(self.root, text="Channels of interest:").place(relx=0.01, rely=0.15)

        self.channel_vars = []
        col = 0.1
        for i in range(self.image.num_channels):
            channel_var = tk.BooleanVar()
            channel_var.set(True)
            self.channel_vars.append(channel_var)

            tk.Label(self.root, text=f"{i+1}:").place(relx=col, rely=0.15)
            col += 0.015
            tk.Checkbutton(self.root, text="", variable=channel_var).place(relx=col, rely=0.15)
            col += 0.015

        SO.channel_vars = self.channel_vars

        tk.Checkbutton(self.root, text="2D", variable=self.segment_2D).place(relx=0.01, rely=0.20)

        tk.Checkbutton(self.root, text="3D", variable=self.segment_3D).place(relx=0.07, rely=0.20)

        # tk.Checkbutton(self.root, text="Same for all images?", variable=self.same_for_all_images_var).place(relx=0.01, rely=0.40)

        tk.Button(self.root, text="Segment and Process", command=self.segment_and_process).place(relx=0.01, rely=0.30, relwidth=0.2, relheight=0.1)

        tk.Label(self.root, text="Channels: (top-to-bottom, left-to-right) 0 -> n").place(relx=0.33, rely=0.01)
        mip_panel = ImageTk.PhotoImage(Image.fromarray(cv2.resize(np.array(self.image.mip_panel).astype('uint8'), (int(0.35 * window_size[0]), int(0.9 * window_size[1])))))
        self.mip_panel_label = tk.Label(self.root, image=mip_panel)
        self.mip_panel_label.place(relx=0.25, rely=0.05)

        tk.Label(self.root, text="Mask channel").place(relx=0.78, rely=0.01)
        mip = ImageTk.PhotoImage(Image.fromarray(cv2.resize(self.image.mip[0, :, :].astype('uint8'), (int(0.35 * window_size[0]), int(0.9 * window_size[1])))))
        self.mask_label = tk.Label(self.root, image=mip)
        self.mask_label.place(relx=0.64, rely=0.05)

        self.mask_channel_var.trace('w', self.mask_channel_changed)

        self.root.mainloop()

    def wait_and_get_values(self):
        self.root.wait_window()
        return {
            'mask_channel': self.mask_channel_var.get(),
            'cyto': self.cyto_var.get(),
            'nuc': self.nuc_var.get(),
            'channels_of_interest': [var.get() for var in self.channels_of_interest_vars],
            'same_for_all_images': self.same_for_all_images_var.get()
        }

    def toggle_cyto(self):
        if self.nuc_var.get():
            self.cyto_var.set(False)
        else:
            self.cyto_var.set(True)

        SO.cyto_var = self.cyto_var.get()

    def toggle_nuc(self):
        if self.cyto_var.get():
            self.nuc_var.set(False)
        else:
            self.nuc_var.set(True)

        SO.nuc_var = self.nuc_var.get()

    def mask_channel_changed(self, *args):
        mask_channel = self.mask_channel_var.get()
        mask = self.image.mip[mask_channel, :, :]
        mask_resized = cv2.resize(mask, (500, 500))
        self.mask_image = ImageTk.PhotoImage(Image.fromarray(mask_resized))
        self.mask_label.config(image=self.mask_image)
        self.mask_label.image = self.mask_image

        SO.mask_channel = mask_channel

    def segment_and_process(self):
        self.channels_of_interest_vars = [var for var in self.channel_vars if var.get()]
        SO.channels_of_interest_vars = self.channels_of_interest_vars
        self.root.destroy()

def reduction(obj, ufunc, method, axis, dtype, out):
    return ufunc.reduce(obj, axis, dtype, out)

def xml_to_dict(xml_str):
    def recursive_dictify(element):
        children = list(element)
        if not children:
            return element.text
        return {child.tag: recursive_dictify(child) for child in children}

    root = ET.fromstring(xml_str)
    return recursive_dictify(root)

def ends_with_ch0X_tif(filename):
    # The regular expression pattern to match the filename
    pattern = r'_ch0[0-9]\.tif$'

    # Use the search function to check if the pattern matches the end of the filename
    match = re.search(pattern, filename)

    # If match is not None, that means the pattern was found in the filename
    return match is not None

def find_all_images_in_path(search_path, filetype=''):
    """
    search a given directory for all images czi, lif, tif
    """
    os.chdir(search_path)
    image_files = []
    for root, _, files in os.walk(search_path):
        for file in files:
            if file[0] == '.':
                continue # since mac folder info
            if file.endswith(filetype):
                if ends_with_ch0X_tif(file):
                    continue
                image_files.append(os.path.join(root, file))

    if not image_files:
        tk.messagebox.showerror('Python Error', 'no images found in search path')
        return

    return image_files


def process_mosaic(m, mosaic, cellpose_model, diam, flow_threshold, cellprob_threshold, channels):

    print(f'Processing mosaic {m}')
    cellpose_out = cellpose_model.eval(
        np.max(mosaic[:, :, :, :, m], axis=1).astype('uint8'),
        diameter=diam,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        channels=channels,
        tile=True
    )

    # Return the necessary data for further processing
    return m, cellpose_out[0], cellpose_out[1][0]

def process_file(po):
    """
    processes a file
    """

    global SO,\
        model_choice,\
        diam, \
        flow_threshold, \
        cellprob_threshold, \
        alter_mask_channel, \
        mask_ch, \
        image

    print('\033[93m', end='')
    print(f'\nprocessing path: {po.image_path}\n'
          f'with save path: {po.save_path}; '
          f'and save options: \n'
          f'channels: {po.save_mip_channels}'
          f', panel: {po.save_mip_panel}'
          f', merge: {po.save_mip_merge}'
          f', dye overlay: {po.save_dye_overlaid}'
          f', colors: {po.save_colors}')
    print('\033[0m', end='')

    image = BioImage()  # initiate

    image.path = po.image_path

    if po.image_path.endswith('.czi'):
        if image.load_czi() == 'metadata_only':
            print(f'Image contained metadata only, skipping: {po.image_path}')
            return
    elif po.image_path.endswith('.lif'):
        image.load_lif()
    elif po.image_path.endswith('.tif'):
        image.load_tif()

    if image.im == [0, 0] and not image.big_image:
        print(f"image empty: {image.path}, skipping")
        return

    # loading and saving the image
    image.extract_colors()

    image.project_mip(side_projections=True, z_scale=3)

    image.normalize(gamma=1)

    image.save(po.save_path,
               po.save_mip_channels,
               po.save_mip_panel,
               po.save_mip_merge,
               po.save_dye_overlaid,
               po.save_colors)

    if not po.segment_image:
        return

    # check if variable 'so' exists
    if 'SO' not in globals():
        # create segmentation GUI
        SegmentationGUI(tk.Tk(), image)

    # if this is still empty but you're segmenting, then assume you want every channel
    if not SO.channels_of_interest_vars:
        SO.channels_of_interest_vars = [True] * image.num_channels

    # if 'model_choice' is not defined or is None, then set its value based on SO.nuc_var
    using_pretrained_model = False
    if not globals().get('model_choice'):
        if SO.nuc_var:
            model_choice = "nuclei"  # @param ["cyto", "nuclei", "cyto2", "tissuenet", "livecell"]
        else:
            model_choice = "cyto"
    else:
        using_pretrained_model = True

    image.mask_ch = SO.mask_channel_var
    if using_pretrained_model:
        cellpose_model = models.CellposeModel(gpu=True, pretrained_model=model_choice)
    else:
        cellpose_model = models.Cellpose(gpu=True, model_type=model_choice)

    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        cellpose_model.device = device("mps")

    if globals().get('alter_mask_channel'):
        if len(mask_ch) == 1:
            channels = [image.mask_ch, image.mask_ch]
        elif len(mask_ch) == 2:
            channels = mask_ch

        image.mask_ch = mask_ch[0]

    else:
        # image.mask_ch was fine before you started messing with this.
        channels = [[image.mask_ch, image.mask_ch]]

    if not ('diam' in globals()):
        diam = None  # for autofit

    # diam = 7 * image.scale[1]  # ~7 µm diameter for hiPSCs
    if not ('flow_threshold' in globals()):
        flow_threshold = 0.3  # @{type:"slider", min:0.1, max:1.1, step:0.1}

    if not ('cellprob_threshold' in globals()):
        cellprob_threshold = -1.5  # @{type:"slider", min:-6, max:6, step:1}

    if not ('stitch_threshold' in globals()):
        stitch_threshold = 0.0  # @{type:"slider", min:-6, max:6, step:1}

    if SO.segment_2D or image.im.shape[1] == 1:

        if not image.big_image:

            print('segmenting 2D image / mip ...')

            # reorganize to a rgb
            cellpose_image = np.zeros((3, image.mip.shape[2], image.mip.shape[1]))

            ''' number on the RHS is the channel number from the image name (i.e. X in _ch0X.tif) '''
            cellpose_image[0, :, :] = image.mip[2, :, :]  # R
            cellpose_image[1, :, :] = image.mip[1, :, :]  # G
            cellpose_image[2, :, :] = image.mip[0, :, :]  # B

            # note that image.mask_ch is not transformed with the cellpose bindings

            # cellpose_image = cellpose_image.transpose(2, 1, 0).astype('uint8')
            # plt.imshow(cellpose_image)
            # print(f'diam:{diam}, ft:{flow_threshold}, cpt:{cellprob_threshold}, ch:{channels}')
            # plt.savefig('out.png')

            cellpose_out = cellpose_model.eval(cellpose_image.astype('uint8'),
                                        diameter=diam,
                                        flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold,
                                        channels=channels)

            image.mask = cellpose_out[0]
            image.flow = cellpose_out[1]
            try:
                image.diam = cellpose_out[3]
            except:
                print('no diameter returned')
                image.diam = -1

        if image.big_image:

            image.mask = np.zeros((image.mosaic.shape[2], image.mosaic.shape[3], image.num_mosaics))
            image.flow = np.zeros((image.mosaic.shape[2], image.mosaic.shape[3], 3, image.num_mosaics))
            image.masks_combined = np.zeros(
                            (image.bbox[:, 1].max() + image.bbox[:, 3].max(),
                             image.bbox[:, 0].max() + image.bbox[:, 2].max(),
                             4), dtype=image.mask.dtype)

            """number on the RHS is the channel number from the image name
            (i.e. X in _ch0X.tif), in order RGB"""

            image.mosaic = image.mosaic[[2, 1, 0], :, :, :, :]

            # # parallel -> untested, something like this might work
            # # Create a partial function with the fixed parameters
            # partial_process_mosaic = partial(
            #     process_mosaic,
            #     mosaic=image.mosaic,
            #     cellpose_model=cellpose_model,
            #     diam=diam,
            #     flow_threshold=flow_threshold,
            #     cellprob_threshold=cellprob_threshold,
            #     channels=channels
            # )
            #
            # # Use the partial function with the multiprocessing Pool
            # print('Initializing a parallel process...')
            # pool = Pool(processes=cpu_count())
            # results = list(tqdm(pool.imap(partial_process_mosaic, range(image.num_mosaics)), desc="Segmenting a big mosaic"))
            #
            # # multiprocessing for big images is necessary
            # for result in tqdm(results, desc="Processing results"):
            #     m, mask, flow = result
            #
            #     # Determine the # rows and columns in the mosaic (assumes square im)
            #     row = image.bbox[m, 0] // image.bbox[1, 0]
            #     col = image.bbox[m, 1] // image.bbox[1, 0]
            #
            #     image.mask[:, :, m] = mask
            #     image.flow[:, :, :, m] = flow
            #
            #     # Add the mask to the appropriate z-stack
            #     image.masks_combined[image.bbox[m, 1]:image.bbox[m, 1] + image.bbox[m, 3],
            #                         image.bbox[m, 0]:image.bbox[m, 0] + image.bbox[m, 2],
            #                         (row % 2) * 2 + (col % 2)
            #                         ] = image.mask[:, :, m]

            # serial
            max_in_masks = 0

            for m in tqdm(range(image.num_mosaics), desc="Segmenting a big mosaic"):

                row = image.bbox[m, 0] // image.bbox[1, 0]
                col = image.bbox[m, 1] // image.bbox[1, 0]

                # # here for debugging / testing. check to see if the image is being segmented correctly
                # if row > 2 or col > 2:
                #     continue

                cellpose_image = np.max(image.mosaic[:, :, :, :, m], axis=1)

                cellpose_out = cellpose_model.eval(
                                        cellpose_image.astype('uint8'),
                                        diameter=diam,
                                        flow_threshold=flow_threshold,
                                        cellprob_threshold=cellprob_threshold,
                                        channels=channels,
                                        tile=True)

                image.mask[:, :, m] = cellpose_out[0] + max_in_masks * (cellpose_out[0] > 0)
                image.flow[:, :, :, m] = cellpose_out[1][0]

                # Add the mask to the appropriate z-stack
                image.masks_combined[image.bbox[m, 1]:image.bbox[m, 1] + image.bbox[m, 3],
                                     image.bbox[m, 0]:image.bbox[m, 0] + image.bbox[m, 2],
                                     (row % 2) * 2 + (col % 2)] = image.mask[:, :, m]

                # max of all in image
                max_in_masks = image.masks_combined.max()

            # combine masks handling overlap
            image.mask = image.combine_big_image_masks(image.masks_combined).astype('uint16')

        # process
        image.process_2d_mask()

        # plot
        image.plot_2d_mask()

        # save into a comma separated value file
        image.save_2d_mask(po)

    if SO.segment_3D and image.im.shape[1] > 1:
        print('no implementation for 3D as yet')

    return image.mask_props

def scripting():

    global model_choice, diam, flow_threshold, cellprob_threshold, alter_mask_channel, mask_ch

    po = ProcessOptions()

    files = find_all_images_in_path(r"Z:\PRJ-Lim\Edge Cones", '.tif')
    model_choice = r"Z:\PRJ-Lim\cellseg\Cellpose Models\ARR_CRX_4"

    diam = 14.75  # autofit, or change for purpose
    flow_threshold = 0.9  # defaults ~ 0.3, comment if not neede
    cellprob_threshold = 0.0001  # defaults ~ -1.4?!, comment if not need
    alter_mask_channel = True

    """ I think the inputs here 0-3 are: 0 for gray, 1 for red, 2 for green, 3 for blue """
    mask_ch = [1, 2] # this might accept two channels like this in a list, but a len(1) int should work as well
    #   ch00 = blue = 3
    #   ch01 = green = 2
    #   ch02 = red = 1

    mask_properties = [None] * len(files)

    for n, file in enumerate(files):

        po.image_path = file
        po.save_path = os.path.dirname(file)

        po.save_mip_channels = False
        po.save_mip_panel = False
        po.save_mip_merge = False
        po.save_dye_overlaid = False
        po.save_colors = False

        global SO
        SO = SegmentationOptions()
        po.segment_image = True

        mask_properties[n] = process_file(po)

    # save a csv file with the name of all files, and the total number of mask objects in each image
    print('saving master cell count')
    with open(os.path.join(po.save_path, 'image_count.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'num_mask_objects'])
        files_and_masks = zip(files, mask_properties)
        for file, mask in files_and_masks:
            writer.writerow([file, len(mask.area)])


if __name__ == '__main__':

    # gui = MainGUI()

    scripting()

    # check that GUI still works with the introduction of the new class SO.

    # check out line ~1140 to change the segmentation model for cell pose

    # TBD
    # (1) add 3D segmentation characterisations,
    # (2) add stardist,

    print('Ok.')
