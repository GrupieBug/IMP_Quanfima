import numpy as np
from skimage import filters
from quanfima import morphology as mrph
from quanfima import visualization as vis
from quanfima import utils


def main():
    data = np.memmap('../../data/polymer3d_8bit_128x128x128.raw',
                     shape=(128, 128, 128), dtype=np.uint8, mode='r')

    data_seg = np.zeros_like(data, dtype=np.uint8)
    for i in range(data_seg.shape[0]):
        th_val = filters.threshold_otsu(data[i])
        data_seg[i] = (data[i] > th_val).astype(np.uint8)

    # estimate porosity
    pr = mrph.calc_porosity(data_seg)
    for k, v in pr.items():
        print('Porosity ({}): {}'.format(k, v))

    # prepare data and analyze fibers
    pdata, pskel, pskel_thick = utils.prepare_data(data_seg)
    oprops = mrph.estimate_tensor_parallel('polymer_orientation_w32', pskel, pskel_thick, 32, '../../data/results')

    odata = np.load(oprops['output_path']).item()
    lat, azth, skel = odata['lat'], odata['azth'], odata['skeleton']

    dprops = mrph.estimate_diameter_single_run('polymer_diameter',
                                               '../../data/results',
                                               pdata, skel, lat, azth)
    dmtr = np.load(dprops['output_path']).item()['diameter']
# plot results
    vis.plot_3d_orientation_map('polymer_w32', lat, azth,
                                output_dir='../../data/results',
                                camera_azth=40.47,
                                camera_elev=32.5,
                                camera_fov=35.0,
                                camera_loc=(40.85, 46.32, 28.85),
                                camera_zoom=0.005124)

    vis.plot_3d_diameter_map('polymer_w32', dmtr,
                             output_dir='../../data/results',
                             measure_quantity='vox',
                             camera_azth=40.47,
                             camera_elev=32.5,
                             camera_fov=35.0,
                             camera_loc=(40.85, 46.32, 28.85),
                             camera_zoom=0.005124,
                             cb_x_offset=5,
                             width=620)


if __name__ == "main":
    main()
