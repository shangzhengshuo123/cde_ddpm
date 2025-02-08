from fid_helper_pytorch import FidHelper

fidhelper = FidHelper(resample_mode='pil_bicubic_float')

fake_img_dir = 'D:/Pythoncode/Dataset/English/213123/fake'
real_img_dir = 'D:/Pythoncode/Dataset/English/213123/real'

# Compute real image stat from image folder.
fidhelper.compute_ref_stat_from_dir(fake_img_dir, batch_size=16, num_workers=2, verbose=True)

# Compute fake image stat from image folder.
fidhelper.compute_eval_stat_from_dir(real_img_dir, batch_size=24, num_workers=3, verbose=True)

# Compute fid.
fid_score = fidhelper.compute_fid_score()
print('FID:', fid_score)
