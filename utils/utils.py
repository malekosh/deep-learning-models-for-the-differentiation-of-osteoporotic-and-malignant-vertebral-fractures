
import os
import numpy as np
import nibabel as nib
import json
# from .utilities import *
import json
import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
from copy import deepcopy

def reorient_to(img, axcodes_to=('P', 'I', 'R'), verb=False):
    # Note: nibabel axes codes describe the direction not origin of axes
    # direction PIR+ = origin ASL
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    if verb:
        print("[*] Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return newimg


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3, cval=-1024,verb=False):
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(np.rint([
        shp[0] * zms[0] / voxel_spacing[0],
        shp[1] * zms[1] / voxel_spacing[1],
        shp[2] * zms[2] / voxel_spacing[2]
        ]).astype(int))
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=cval)
    if verb:
        print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_mask_to(msk, to_img):
    to_img.header['bitpix'] = 8
    to_img.header['datatype'] = 2  # uint8
    new_msk = nib.processing.resample_from_to(msk, to_img, order=0)
    print("[*] Mask resampled to image size:", new_msk.header.get_data_shape())
    return new_msk


def load_centroids(ctd_path):
    with open(ctd_path) as json_data:
        dict_list = json.load(json_data)
        json_data.close()
    if isinstance(dict_list,dict):
        if 'vertebralCentroids' in dict_list.keys():
            dict_list = dict_list['vertebralCentroids']
    ctd_list = []
    for d in dict_list:
        if 'direction' in d:
            ctd_list.append(tuple(d['direction']))
        elif 'nan' in str(d):            #skipping NaN centroids
            continue
        else:
            ctd_list.append([d['label'], d['X'], d['Y'], d['Z']]) 
    return ctd_list

def reorient_centroids_to(ctd_list, img, decimals=1, verb=False):
    # reorient centroids to image orientation
    # todo: reorient to given axcodes (careful if img ornt != ctd ornt)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    if len(ctd_arr) == 0:
        print("[#] No centroids present") 
        return ctd_list
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ornt_fr = nio.axcodes2ornt(ctd_list[0])  # original centroid orientation
    axcodes_to = nio.aff2axcodes(img.affine)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    trans = nio.ornt_transform(ornt_fr, ornt_to).astype(int)
    perm = trans[:, 0].tolist()
    shp = np.asarray(img.dataobj.shape)
    ctd_arr[perm] = ctd_arr.copy()
    for ax in trans:
        if ax[1] == -1:
            size = shp[ax[0]]
            ctd_arr[ax[0]] = np.around(size - ctd_arr[ax[0]], decimals)
    out_list = [axcodes_to]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Centroids reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to)
    return out_list


def rescale_centroids(ctd_list, img, voxel_spacing=(1, 1, 1), verb=False):
    # rescale centroid coordinates to new spacing in current x-y-z-orientation
    ornt_img = nio.io_orientation(img.affine)
    ornt_ctd = nio.axcodes2ornt(ctd_list[0])
    if np.array_equal(ornt_img, ornt_ctd):
        zms = img.header.get_zooms()
    else:
        ornt_trans = nio.ornt_transform(ornt_img, ornt_ctd)
        aff_trans = nio.inv_ornt_aff(ornt_trans, img.dataobj.shape)
        new_aff = np.matmul(img.affine, aff_trans)
        zms = nib.affines.voxel_sizes(new_aff)
    ctd_arr = np.transpose(np.asarray(ctd_list[1:]))
    v_list = ctd_arr[0].astype(int).tolist()  # vertebral labels
    ctd_arr = ctd_arr[1:]
    ctd_arr[0] = np.around(ctd_arr[0] * zms[0] / voxel_spacing[0], decimals=1)
    ctd_arr[1] = np.around(ctd_arr[1] * zms[1] / voxel_spacing[1], decimals=1)
    ctd_arr[2] = np.around(ctd_arr[2] * zms[2] / voxel_spacing[2], decimals=1)
    out_list = [ctd_list[0]]
    ctd_list = np.transpose(ctd_arr).tolist()
    for v, ctd in zip(v_list, ctd_list):
        out_list.append([v] + ctd)
    if verb:
        print("[*] Rescaled centroid coordinates to spacing (x, y, z) =", voxel_spacing, "mm")
    return out_list

def crop_slice(msk, dist=20):
    shp = msk.dataobj.shape
    zms = msk.header.get_zooms()
    d = np.around(dist / np.asarray(zms)).astype(int)
    msk_bin = np.asanyarray(msk.dataobj, dtype=bool)
    msk_bin[np.isnan(msk_bin)] = 0
    cor_msk = np.where(msk_bin > 0)
    c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
    x0 = c_min[0]-d[0] if (c_min[0]-d[0]) > 0 else 0
    y0 = c_min[1]-d[1] if (c_min[1]-d[1]) > 0 else 0
    z0 = c_min[2]-d[2] if (c_min[2]-d[2]) > 0 else 0
    x1 = c_max[0]+d[0] if (c_max[0]+d[0]) < shp[0] else shp[0]
    y1 = c_max[1]+d[1] if (c_max[1]+d[1]) < shp[1] else shp[1]
    z1 = c_max[2]+d[2] if (c_max[2]+d[2]) < shp[2] else shp[2]
    ex_slice = tuple([slice(x0, x1), slice(y0, y1), slice(z0, z1)])
    origin_shift = tuple([x0, y0, z0])
    return ex_slice, origin_shift


def crop_centroids(ctd_list, o_shift):
    for v in ctd_list[1:]:
        v[1] = v[1] - o_shift[0]
        v[2] = v[2] - o_shift[1]
        v[3] = v[3] - o_shift[2]
    return ctd_list

def get_derivatives_from_rawdata(img_pth):
    ctd_pth = img_pth.replace('rawdata', 'derivatives').replace('_ct.nii.gz', '_seg-subreg_ctd.json')
    msk_pth = img_pth.replace('rawdata', 'derivatives').replace('_ct.nii.gz', '_seg-vert_msk.nii.gz')
    return ctd_pth, msk_pth, 

def get_img_paths(directory, ce_in_path=False):
    '''
    Returns a list of ct paths from a BIDS structured dataset
    optional variable ce_in_path to be used to strictly return paths with ce-label in name 
    '''
    im_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith("_ct.nii.gz"):
                if ce_in_path:
                    if '_ce-' in file:
                        im_paths.append(os.path.join(root, file))
                else:
                    im_paths.append(os.path.join(root, file))
    return im_paths

def get_min_max_pad(pos, size, s=32):
    if pos - s > 0:
        pos_min  = pos - s
        pad_min = 0
    else:
        pos_min = 0
        pad_min = s-pos
    if pos + s< size:
        pos_max = pos + s
        pad_max = 0
    else:
        pos_max = size
        pad_max = pos + s - size
    return pos_min, pos_max, pad_min, pad_max
def get_3d_data(img_data, ctd, verts):
    s = 35
    size = img_data.shape
    vertebrae_dict = {}
    for ct in ctd[1:]:
        if ct[0] in verts:
            x,y,z = int(round(ct[1])), int(round(ct[2])), int(round(ct[3]))
            if ct[0] > 1:
                x_min, x_max, x_pad_min, x_pad_max = get_min_max_pad(x, size[0], s)
                y_min, y_max, y_pad_min, y_pad_max = get_min_max_pad(y, size[1], s)
                z_min, z_max, z_pad_min, z_pad_max = get_min_max_pad(z, size[2], s)


                cropped_vert = img_data[x_min:x_max, y_min:y_max, z_min:z_max]
                avg = np.mean(cropped_vert)
                padded_vert = np.pad(cropped_vert, ((x_pad_min,x_pad_max),(y_pad_min,y_pad_max),(z_pad_min, z_pad_max)),constant_values=((avg,avg),(avg,avg),(avg,avg)))


                vertebrae_dict[ct[0]] = {'im':padded_vert}

        
    return vertebrae_dict
        
def read_datapoint(img_pth):
    to_ax = ('I', 'A', 'L')
    ctd_pth, msk_pth = get_derivatives_from_rawdata(img_pth)
    
    if not os.path.isfile(ctd_pth):
        print('No ctd file found for {}'.format(img_pth))
        return None
    
    
    img = nib.load(img_pth)
    ctd = load_centroids(ctd_pth)
    
    if os.path.isfile(msk_pth):
        msk = nib.load(msk_pth)
        ex_slice, o_shift = crop_slice(msk,10)
    
        img = img.slicer[ex_slice]
        msk = msk.slicer[ex_slice]
        ctd = crop_centroids(ctd, o_shift)

    img = reorient_to(img, to_ax)
    ctd = reorient_centroids_to(ctd, img)

    ctd = rescale_centroids(ctd, img, (1, 1, 1))
    img = resample_nib(img, voxel_spacing=(1, 1, 1), order=3)

    img_data = (np.clip(np.asanyarray(img.dataobj,dtype=np.float64),-1000,3500) +1000)/2500
    
    return img_data, ctd
