import os
import zipfile

root_dir = '/media/akosta/Datasets/KITTI_raw/'
tgt_dir = '/media/akosta/Datasets/KITTI_processed/'


zip_files = sorted([fname for fname in os.listdir(root_dir)])

for zip_name in zip_files:
    if 'sync' in zip_name:
        print(zip_name)
        with zipfile.ZipFile(os.path.join(root_dir, zip_name), 'r') as zip_ref:
            zip_ref.extractall(tgt_dir)


#%%
src_dir = '/media/akosta/Datasets/KITTI_processed/train/'
tgt_dir = '/media/akosta/Datasets/KITTI_processed/test/left/'

files = sorted([fname for fname in os.listdir(tgt_dir)])

for i in range(len(files)):
    os.rename(os.path.join(tgt_dir, files[i]), os.path.join(tgt_dir, 'test_' + str(i) + '.png'))

# %%
