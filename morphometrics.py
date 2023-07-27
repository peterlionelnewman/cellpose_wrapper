import glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_parameter_histogram(folder_path, par, ax):
    # Get list of all CSV files in the given directory
    csv_files = glob.glob(folder_path + "/*.csv")

    # Create an empty DataFrame
    data = pd.DataFrame()

    # Read each csv file and concatenate into the data DataFrame
    for file in csv_files:
        df = pd.read_csv(file)
        data = pd.concat([data, df])

    # clean data
    data = data[data['minor_ax'] != 0]  # this shouldn't exist
    data = data[data['area'] >= 80]  # these cells are suspiciously small
    data = data[data['area'] <= 600]  # these cells are suspiciously big

    # create aspect ratio
    data['aspect_ratio'] = data['major_ax'] / data['minor_ax']

    # Check if the parameter exists in the DataFrame columns
    if par not in data.columns:
        print(f"Parameter {par} not found in the data.")
        return

    # Plot histogram
    ax.hist(data[par], bins=30, edgecolor='black')
    ax.set_xlabel(par)
    ax.set_ylabel('Frequency')
    # ax.set_yscale('log')


def overlay_parameter_on_image(csv_file, par, ax):
    # Read the data from the CSV file
    df = pd.read_csv(csv_file)

    # Clean data
    df = df[df['minor_ax'] != 0]  # this shouldn't exist
    df = df[df['area'] >= 80]  # these cells are suspiciously small
    df = df[df['area'] <= 600]  # these cells are suspiciously big

    # Create aspect ratio
    df['aspect_ratio'] = df['major_ax'] / df['minor_ax']

    # Check if the parameter exists in the DataFrame columns
    if par not in df.columns:
        print(f"Parameter {par} not found in the data.")
        return

    # Construct image file path
    image_file = csv_file.replace("_2Dresults.csv", "")

    # Load the image
    image = mpimg.imread(image_file)

    im_gray = image.mean(axis=2)

    # Show the image
    ax.imshow(im_gray, cmap=plt.cm.gray)

    # Create scatter plot overlay
    magic_number = 0.85
    s = df[par].to_numpy()
    s = s / s.max()
    ind = np.argwhere(s > magic_number).squeeze()
    scatter = ax.scatter(df['centroid_y'].to_numpy()[ind], df['centroid_x'].to_numpy()[ind], s=s[ind], c='magenta', alpha=0.5)
    ind = np.argwhere(s <= magic_number).squeeze()
    scatter = ax.scatter(df['centroid_y'].to_numpy()[ind], df['centroid_x'].to_numpy()[ind], s=s[ind], c='cyan', alpha=0.5)

    # Set the title of the plot to the image file name
    ax.set_title(image_file)

    # Return the scatter plot (in case further customization is needed)
    return scatter


def plot_2d_histogram(csv_file, par1, par2, ax):
    # Read the data from the CSV file
    df = pd.read_csv(csv_file)

    # Clean data
    df = df[df['minor_ax'] != 0]  # this shouldn't exist
    df = df[df['area'] >= 80]  # these cells are suspiciously small
    df = df[df['area'] <= 600]  # these cells are suspiciously big

    # Create aspect ratio
    df['aspect_ratio'] = df['major_ax'] / df['minor_ax']

    # Check if the parameters exist in the DataFrame columns
    if par1 not in df.columns or par2 not in df.columns:
        print(f"One of the parameters {par1} or {par2} not found in the data.")
        return

    # Plot 2D histogram
    hb = ax.hist2d(df[par1], df[par2], bins=20, cmap='hot')
    ax.set_xlabel(par1)
    ax.set_ylabel(par2)

    # Colorbar
    cb = plt.colorbar(hb[3], ax=ax)
    cb.set_label('counts in bin')


if __name__ == '__main__':

    parameters = ['area', 'perimeter', 'major_ax', 'minor_ax', 'aspect_ratio', 'eccentricity', 'convexity']

    # ### histograms of pars
    # fig, ax = plt.subplots(len(parameters), 2, figsize=(10, 20), dpi=200)
    #
    # for n, parameter in enumerate(parameters):
    #
    #     path = '/Users/peternewman/Drive/Python/plot/BL morphometrics/timepoint a'
    #     if n == 0:
    #         ax[n, 0].set_title(f"{Path(path).name}")
    #     plot_parameter_histogram(path, parameter, ax[n, 0])
    #
    #     path = '/Users/peternewman/Drive/Python/plot/BL morphometrics/timepoint b'
    #     if n == 0:
    #         ax[n, 1].set_title(f"{Path(path).name}")
    #     plot_parameter_histogram(path, parameter, ax[n, 1])
    #
    # plt.savefig('histograms.png')
    # plt.show()

    # ### image of pars
    # folder_path = '/Users/peternewman/Drive/Python/plot/BL morphometrics/timepoint a'
    # csv_files = glob.glob(folder_path + "/*.csv")
    #
    # sqrt_len = np.ceil(len(csv_files) ** 0.5).astype(int)
    #
    # for par in parameters:
    #     fig, ax = plt.subplots(sqrt_len, sqrt_len, figsize=(20, 20), dpi=80)
    #
    #     for n, file in enumerate(csv_files):
    #         n, m = np.unravel_index(n, ax.shape)
    #         overlay_parameter_on_image(file, par, ax[n, m])
    #
    #     plt.savefig(f'{folder_path[-1]}_scatter_overlaid_{par}.png')
    #     plt.show()

    # ### image of pars
    # folder_path = '/Users/peternewman/Drive/Python/plot/BL morphometrics/timepoint b'
    # csv_files = glob.glob(folder_path + "/*.csv")
    #
    # sqrt_len = np.ceil(len(csv_files) ** 0.5).astype(int)
    #
    # for par in parameters:
    #     fig, ax = plt.subplots(sqrt_len, sqrt_len, figsize=(20, 20), dpi=80)
    #
    #     for n, file in enumerate(csv_files):
    #         n, m = np.unravel_index(n, ax.shape)
    #         overlay_parameter_on_image(file, par, ax[n, m])
    #
    #     plt.savefig(f'{folder_path[-1]}_scatter_overlaid_{par}.png')
    #     plt.show()

    ## 2D histograms
    csv_file = '/Users/peternewman/Drive/Python/plot/BL morphometrics/timepoint a/20x FGF8 RA- 1_Processed001.tif_2Dresults.csv'
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    plot_2d_histogram(csv_file, 'major_ax', 'eccentricity', ax)
    plt.savefig('2D_histogram.png')
    plt.show()