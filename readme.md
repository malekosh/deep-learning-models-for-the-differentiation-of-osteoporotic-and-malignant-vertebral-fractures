# Development and validation of CT-based deep learning models for the differentiation of osteoporotic and malignant vertebral fractures

This repository contains the code for distinguishing between malignat and osteoporotic fractures in CT scans in the nifti format.

### Data Format

* The dataset has to have three files corresponding to one data sample: image, segmentation mask, centroid annotations. The format of those files follows  **VerSe dataset** format (https://github.com/anjany/verse) 

* Sub-directory-based arrangement for each patient. File names are constructed of entities, a suffix and a file extension following the conventions of the Brain Imaging Data Structure (BIDS; https://bids.neuroimaging.io/)

```
Example:
-------
sample_dataset/rawdata/sub-sample000
    sub-sample000_ct.nii.gz - CT image series

sample_dataset/derivatives/sub-sample000/
    sub-sample000_seg-vert_msk.nii.gz - Segmentation mask of the vertebrae
    sub-sample000_seg-subreg_ctd.json - Centroid coordinates in image space

```
## Usage

```
python3 script.py  -path 1  -vert L2 L3
```


## References:
<ol>

<li>
.
</li>

</ol>


## *Under Construction*
- [x] Push inference code
- [] Batch input support
- [] Push training and test code
- [] Update readme
