
# Common Libraries
import os, nltools, nilearn
import numpy as np
import pandas as pd
import nibabel as nb
from nltools.data import Brain_Data
from nilearn.plotting import plot_roi
from nilearn.image import resample_to_img, math_img, mean_img

# Custom Libraries
from dictionary_util import search_dict

def apply_mask_change_shape(fMRI_datas, mask):
    """
    Convert fMRI data(3d) applied mask to 1d data

    :param fMRI_datas: fMRI_datas(list - nitfti image)
    :param mask: nifti_mask

    return masked_fmri_datas(1d array), mask_img
    """
    resampled_mask = resample_to_img(mask, 
                                     fMRI_datas[0], 
                                     interpolation = "nearest", 
                                     force_resample = True,
                                     copy_header = True)
    resampled_mask = nb.Nifti1Image(np.array(resampled_mask.get_fdata() > 0, dtype=np.int8), resampled_mask.affine)

    # Multiply the functional image with the mask
    roi_fMRI_datas = []
    for target_data in fMRI_datas:
        roi_img = nilearn.masking.apply_mask(target_data, resampled_mask)
        roi_fMRI_datas.append(roi_img)

    return roi_fMRI_datas, resampled_mask

def apply_mask_no_change_shape(fMRI_datas, mask):
    """
    While preserving shape, apply ROI mask to fMRI datas

    :params fMRI_datas: array of Nifti1Image(list)
    :params mask: Nifti1Image (no need to fit shape to fMRI_datas)

    return fMRI_dats(list), mask_img
    """
    resampled_mask = resample_to_img(mask, fMRI_datas[0], interpolation="nearest")
    resampled_mask = nb.Nifti1Image(np.array(resampled_mask.get_fdata() > 0, dtype=np.int8), resampled_mask.affine)

    # Multiply the functional image with the mask
    roi_fMRI_datas = []
    for target_data in fMRI_datas:
        if len(target_data.shape) == len(resampled_mask.shape):
            roi_img = math_img('img1 * img2', img1=target_data, img2=resampled_mask)
            roi_fMRI_datas.append(roi_img)
        else:
            roi_img = math_img('img1 * img2', img1=target_data, img2=resampled_mask.slicer[..., None])
            roi_fMRI_datas.append(roi_img)

    return roi_fMRI_datas, resampled_mask

def apply_mask_withCrop(fMRI_datas, mask):
    """
    While removing zeros, apply ROI mask to fMRI datas

    :params fMRI_datas: array of Nifti1Image(list)
    :params mask: Nifti1Image (no need to fit shape to fMRI_datas)

    return fMRI_dats(list), mask_img
    """
    roi_datas, resampled_mask = apply_mask_no_change_shape(fMRI_datas, mask)

    # Remove assplit_data_pairs many zero rows in the data matrix to reduce overall volume size
    from nilearn.image import crop_img

    roi_crop_fMRI_datas = []
    for roi_data in roi_datas:
        roi_crop_fMRI_datas.append(crop_img(roi_data))
    return roi_crop_fMRI_datas, resampled_mask

def apply_mask_with_img(anatomy_data, fMRI_datas, mask, is_show_img = True):
    """
    Applying mask with showing the result of mask application
    
    :params anatomy_data: anatomy(nifiti1Image)
    :params fMRI_datas: array of Nifti1Image(list)
    :params mask: Nifti1Image (no need to fit shape to fMRI_datas)
    
    return fMRI_dats(list), mask_img
    """
    if len(fMRI_datas[0].shape) == 3:
        reference_img = fMRI_datas[0]
    else:
        reference_img = mean_img(fMRI_datas[0])
    resampled_anatomy = resample_to_img(anatomy_data, 
                                        reference_img, 
                                        force_resample = True,
                                        copy_header = True)
    resampled_mask = resample_to_img(mask, 
                                     fMRI_datas[0], 
                                     interpolation="nearest", 
                                     force_resample = True,
                                     copy_header = True)
    
    if is_show_img:
        plot_roi(resampled_mask,
                 black_bg=True,
                 bg_img=resampled_anatomy,
                 cut_coords=None,
                 cmap='magma_r',
                 dim=1)
    return apply_mask_change_shape(fMRI_datas, mask)

class brain_mask:
    def __init__(self, mask_nifti_img, name):
        """
        :param mask_nifti_img: (nifti image) or path
        :param name: mask name(string)
        """
        if type(mask_nifti_img) == str:
            self.mask_nifti_img = nb.load(mask_nifti_img)
        elif type(mask_nifti_img) == nb.Nifti1Image:
            self.mask_nifti_img = mask_nifti_img
        else:
            raise ValueError('mask_nifti_img need to be set nifti image file')

        self.name = name

    def get_data(self):
        return self.mask_nifti_img

    def apply(self, anatomy, fMRI_datas, is_show_img = True):
        """
        Apply Mask to fMRI datas

        :param anatomy: for viewing background
        :param fMRI_datas: nifti(list)
        :param is_show_img: True shows the roi image

        return fmri_datas applied mask (these datas' shape is changed by mask)
        """

        image_only = 0
        return apply_mask_with_img(anatomy_data = anatomy,
                                   fMRI_datas = fMRI_datas,
                                   mask = self.mask_nifti_img,
                                   is_show_img = is_show_img)[image_only]

class fan_roi_mask_manager:
    def __init__(self, fan_info_path, mask_dir_path, reference_img):
        """
        :params fan_info_path: fan roi path
        :params reference_img: reference_img for fitting shape
        """
        if type(reference_img) == str:
            reference_img = nb.load(reference_img)
        self.reference_img = reference_img
        
        # load fan info
        self.mask_fan_info = pd.read_csv(fan_info_path, header=None)
        self.mask_fan_info.index = self.mask_fan_info.index + 1 # roi file_name is started from 1
        self.mask_fan_info.index = list(map(lambda index: str(index).zfill(3), self.mask_fan_info.index)) # for matching filename format ex) 001
        self.mask_fan_info.columns = ["Description"]
        self.mask_fan_info = dict(self.mask_fan_info["Description"])
    
        self.mask_dir_path = mask_dir_path
        
    def search_mask_info(self, keywords):
        """
        search roi information using keywords
        
        :param keywords: keyword for searching ex) ["Rt", "prefrontal"]
        """
        searched_dict = search_dict(self.mask_fan_info, keywords)
        return searched_dict
    
    def search_roi(self, keywords, exclude_keywords = None):
        """
        :param keywords: keyword to search(list)
        """
        name = "_".join(keywords)
        if exclude_keywords != None:
            name += "_exclude_" + "_".join(exclude_keywords)
            
        return brain_mask(mask_nifti_img = self.make_roi_with_search(keywords = keywords, exclude_keywords = exclude_keywords),
                          name = name)
    
    def make_roi_with_search(self, keywords, exclude_keywords = None):
        """
        search roi paths and return roi(nifiti img)

        :params dict_roi_info: roi info dictionary
        :params keywords: search keywords ex ["precentral gyrus", "subiculum"]
        :params reference_img: reference_img for fitting shape

        return: roi(nifiti img)
        """
        searched_dict = search_dict(self.mask_fan_info, keywords)
        search_paths = [os.path.join(self.mask_dir_path, "fan.roi." + key + ".nii.gz") for key in searched_dict] 

        # exclude
        if exclude_keywords != None:
            exclude_searched_dict = search_dict(self.mask_fan_info, exclude_keywords)
            exclude_search_paths = [os.path.join(self.mask_dir_path, "fan.roi." + key + ".nii.gz") for key in exclude_searched_dict] 
        else:
            exclude_search_paths = []
        
        # filter
        filtered_paths = []
        for search_path in search_paths:
            if search_path not in exclude_search_paths:
                filtered_paths.append(search_path)
                
        return make_roi(filtered_paths, reference_img = self.reference_img)

class parcellation_roi_mask_manager:
    def __init__(self, parcellation_info_path, mask_path, reference_img):
        """
        :params parcellation_info_path: parcellation roi info path
        :params mask_path: mask path
        :params reference_img: reference_img for fitting shape
        """
        self.mask_info = self.load_roi_info(parcellation_info_path)
        self.mask_path = mask_path
        self.reference_img = reference_img
        
    def load_roi_info(self, parcellation_info_path):
        """
        load roi information
        
        :params parcellation_info_path: parcellation roi info path
        return { roi description : index }
        """
        with open(parcellation_info_path) as f:
            lines = f.readlines()
        
        parcellation_lines = sj_higher_function.list_map(lines[1:], lambda s: s.strip())
        parcellation_lines = sj_higher_function.list_map(parcellation_lines, lambda s: s.split("="))
        
        parcellation_info = {}
        for key, description in parcellation_lines:
            parcellation_info[int(key)] = description
        
        return parcellation_info
        
    def search_mask_info(self, keywords):
        """
        search roi information using keywords
        
        :param keywords: keyword for searching ex) ["Rt", "prefrontal"]
        """
        searched_dict = search_dict(self.mask_info, keywords)
        return searched_dict
    
    def make_roi_with_search(self, 
                             keywords):
        """
        search roi paths and return roi(nifiti img)
        
        :params mask_path: mask path
        :params dict_roi_info: roi info dictionary
        :params keywords: search keywords ex ["precentral gyrus", "subiculum"]
        :params reference_img: reference_img for fitting shape

        return: roi(nifiti img)
        """
        logic_to_number = np.vectorize(lambda x: 1 if x == True else 0)
        
        searched_dict = search_dict(self.mask_info, keywords)
        
        parcellation_mask = Brain_Data(self.mask_path)
        parcellation_mask_x = nltools.mask.expand_mask(parcellation_mask)

        local_masks = []
        for key in searched_dict:
            local_mask = parcellation_mask_x[key].to_nifti()
            local_mask_array = logic_to_number(local_mask.get_fdata())

            local_mask = nb.Nifti1Image(logic_to_number(local_mask.get_fdata()), local_mask.affine)

            local_masks.append(local_mask)
    
        roi_img = add_imgs(local_masks)
        roi_img = resample_to_img(roi_img, 
                                  self.reference_img, 
                                  interpolation = "nearest",
                                  force_resample = True,
                                  copy_header = True)

        return roi_img

class multi_label_roi_manager:
    def __init__(self, labeled_img, label_info, reference_img):
        """
        Initialize multi label roi manager
        
        :params labeled_img: nifti image with multi labeled(nifti)
        :params label_info: (dictionary)
        :params reference_img: reference_img (nitfti)
        """
        self.labeled_array = resample_to_img(labeled_img, 
                                             reference_img, 
                                             interpolation = "nearest", 
                                             force_resample = True).get_fdata()
        self.label_info = label_info
        self.reference_img = reference_img
        
    def search_mask_info(self, keywords):
        """
        search roi information using keywords
        
        :param keywords: keyword for searching ex) ["Rt", "prefrontal"]
        """
        searched_dict = search_dict(self.label_info, keywords)
        return searched_dict
    
    def make_roi_with_search(self, 
                             keywords):
        """
        search roi paths and return roi(nifiti img)
    
        :params keywords: search keywords ex ["precentral gyrus", "subiculum"]

        return: roi(nifiti img)
        """
        
        searched_dict = search_dict(self.label_info, keywords)

        local_rois = []
        for key in searched_dict:
            check_value = np.vectorize(lambda x: 1 if x == key else 0) 
            local_roi = nb.Nifti1Image(check_value(self.labeled_array).astype(np.int32), self.reference_img.affine)
            local_rois.append(local_roi)
    
        roi_img = add_imgs(local_rois)
        return roi_img

    def make_roi_usingLabel(self, labels):
        """
        Make roi from labels
    
        :params labels: label ex) [1,2,3]

        return: roi(nifiti img)
        """
        local_rois = []
        for label in labels:
            check_value = np.vectorize(lambda x: 1 if x == label else 0) 
            local_roi = nb.Nifti1Image(check_value(self.labeled_array).astype(np.int32), self.reference_img.affine)
            local_rois.append(local_roi)
    
        roi_img = add_imgs(local_rois)
        return roi_img
        
def load_mask(mask_path, resample_target):
    resampled_mask = resample_to_img(nb.load(mask_path), 
                                     resample_target, 
                                     interpolation = "nearest",
                                     copy_header = True)
    return resampled_mask

def add_imgs(imgs, is_use_path = False):
    """
    Add many image from imgs

    :param imgs: imgs(nitfti image array)

    return nifti image
    """

    if len(imgs) == 1:
        if is_use_path == True:
            img = nb.load(imgs[0])
            return img
        else:
            return imgs[0]

    if is_use_path == True:
        temp_imgs = []
        for i in range(len(imgs)):
            img = nb.load(imgs[i])
            temp_imgs.append(img)
        imgs = temp_imgs

    return math_img("img1 + img2", img1=imgs[0], img2=add_imgs(imgs[1:]))

def join_roi_imgs(imgs):
    """
    join roi images

    :param imgs: array(nifti images or img path)

    return roi image(nifti)
    """
    imgs_ = []
    for img in imgs:
        if type(img) == str:
            img_ = nb.load(img)
        elif type(img) == nb.nifti1.Nifti1Image:
            img_ = img
        else:
            pass
        imgs_.append(img_)

    if len(imgs) == 1:
        return imgs[0]

    return math_img("img1 * img2", img1=imgs[0], img2=join_roi_imgs(imgs[1:]))

def make_roi(roi_paths, reference_img):
    """
    join roi images

    :params roi_paths: roi path(nifiti img)
    :params reference_img: reference_img for fitting shape

    return: roi(nifiti img)
    """
    roi = add_imgs(roi_paths)
    roi = resample_to_img(roi, reference_img, interpolation="nearest", force_resample = True)

    return roi

def untangle_mask_img(mask, select_values = None):
    """
    Untangle brain image using unique value
    
    :param img: (Nifti1Image)
    :param select_values: (list)

    return (list)
        -element: (Nifti1Image)
    """
    
    mask_array = mask.get_fdata()

    if select_values == None:
        uq_values = list(np.unique(mask_array).astype(int))
        uq_values.remove(0)
        select_values = uq_values
    
    select_brains = []
    for select_value in select_values:
        brain_array = np.where(mask_array == select_value, 1.0, 0.0).astype(np.float32)
        select_brains.append(nb.Nifti1Image(brain_array, mask.affine))
    return select_brains

if __name__ == "__main__":
    # fan_roi_mask_manager
    mask_img = nb.load("/mnt/sdb2/seojin/tutorial/preprocessed/full_mask.HP01.nii.gz")
    mask_manager = fan_roi_mask_manager(fan_info_path=mask_fan_info_path, 
                                        mask_dir_path=mask_dir_path, 
                                        reference_img=mask_img)
    
    mask_manager.search_roi(["precentral gyrus", "Lt"])
    mask_manager.search_mask_info(["precentral gyrus", "Lt"])
    
    # parcellation_roi_manager
    parcellation_roi_manager = parcellation_roi_mask_manager(parcellation_info_path = parcellation_info_path, 
                                                             mask_path = os.path.join(mask_dir_path, "parcellation", "Neurosynth_parcellation_k50_2mm.nii.gz"), 
                                                             reference_img= full_mask)


    # brainstem navigator
    mask_img = nb.load("/mnt/sdb2/seojin/tutorial/preprocessed/full_mask.HP01.nii.gz")
    
    roi_brain = "/mnt/sdb2/seojin/mask/BrainstemNavigator/0.9/seojin_MNI_ROI_images/brainstem_nav.nii"
    bn_brain = nb.load(roi_brain)
    label_info_path = "/mnt/sdb2/seojin/mask/BrainstemNavigator/0.9/seojin_MNI_ROI_images/brainstem_nav.tsv"
    bn_df = pd.read_csv(label_info_path, delimiter='\t', header = None)
    bn_df.columns = ["key", "value"]
    bn_info = sj_dictionary.df_to_dict(bn_df)
    roi_manager = multi_label_roi_manager(labeled_img = bn_brain, label_info = bn_info, reference_img = mask_img)
    roi_img = roi_manager.search_mask_info(["C"])

    # untangle img
    brain_imgs = untangle_mask_img("/Users/clmn/Downloads/vedo_vis/model/st_c/Clust_mask.nii")
    
    
    