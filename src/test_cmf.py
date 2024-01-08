import torch
from options_cmf import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os
import sys
import numpy
import pdb
import skimage.io
from skimage.metrics import structural_similarity as ssim
import platform

if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_cmf_pt as common_cmf

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  if opts.gpu >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
      device = torch.device("cuda")
  else:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
      device = torch.device("cpu")

  if opts.result_dir and not os.path.exists(opts.result_dir):
      os.makedirs(opts.result_dir)

  test_data_t, test_data_s, _ = common_cmf.load_test_data(opts.dataroot)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(0)
  model.resume(os.path.join(opts.checkpoint_dir, "best.pth"), train=False)
  model.eval()

  test_st_psnr = numpy.zeros((len(test_data_s), 1), numpy.float32)
  test_ts_psnr = numpy.zeros((len(test_data_t), 1), numpy.float32)
  test_st_ssim = numpy.zeros((len(test_data_s), 1), numpy.float32)
  test_ts_ssim = numpy.zeros((len(test_data_t), 1), numpy.float32)
  test_st_mae = numpy.zeros((len(test_data_s), 1), numpy.float32)
  test_ts_mae = numpy.zeros((len(test_data_t), 1), numpy.float32)
  test_st_list = []
  test_ts_list = []
  msg_detail = ""
  with torch.no_grad():
    for i in range(len(test_data_s)):
      test_st = numpy.zeros(test_data_s[i].shape, numpy.float32)
      test_ts = numpy.zeros(test_data_t[i].shape, numpy.float32)
      used = numpy.zeros(test_data_s[i].shape, numpy.float32)
      for j in range(test_data_s[i].shape[0] - opts.input_dim_a + 1):
        test_patch_s = torch.tensor(numpy.expand_dims(test_data_s[i][j:j + opts.input_dim_a, :, :], 0), device=device)
        test_patch_t = torch.tensor(numpy.expand_dims(test_data_t[i][j:j + opts.input_dim_a, :, :], 0), device=device)

        ret_st = model.test_forward_transfer(test_patch_s, test_patch_t, a2b=True)
        ret_ts = model.test_forward_transfer(test_patch_t, test_patch_s, a2b=False)

        test_st[j:j + opts.input_dim_a, :, :] += ret_st.cpu().detach().numpy()[0]
        test_ts[j:j + opts.input_dim_a, :, :] += ret_ts.cpu().detach().numpy()[0]
        used[j:j + opts.input_dim_a, :, :] += 1

      assert used.min() > 0
      test_st /= used
      test_ts /= used
      
      if opts.result_dir:
        common_cmf.save_nii(test_ts, os.path.join(opts.result_dir, "syn_%d.nii.gz" % i))

      st_psnr = common_metrics.psnr(test_st, test_data_t[i])
      ts_psnr = common_metrics.psnr(test_ts, test_data_s[i])
      st_ssim = ssim(test_st, test_data_t[i], data_range=2.)
      ts_ssim = ssim(test_ts, test_data_s[i], data_range=2.)
      st_mae = abs(common_cmf.restore_hu(test_st) - common_cmf.restore_hu(test_data_t[i])).mean()
      ts_mae = abs(common_cmf.restore_hu(test_ts) - common_cmf.restore_hu(test_data_s[i])).mean()

      test_st_psnr[i] = st_psnr
      test_ts_psnr[i] = ts_psnr
      test_st_ssim[i] = st_ssim
      test_ts_ssim[i] = ts_ssim
      test_st_mae[i] = st_mae
      test_ts_mae[i] = ts_mae
      test_st_list.append(test_st)
      test_ts_list.append(test_ts)

      msg_detail += "  %d  psnr: %f  ssim: %f  mae: %f\n" % (i, ts_psnr, ts_ssim, ts_mae)

  msg = "  test_st_psnr:%f/%f  test_st_ssim:%f/%f  test_ts_psnr:%f/%f  test_ts_ssim:%f/%f  test_ts_mae:%f/%f  test_ts_mae:%f/%f\n" % \
        (test_st_psnr.mean(), test_st_psnr.std(), test_st_ssim.mean(), test_st_ssim.std(),
         test_ts_psnr.mean(), test_ts_psnr.std(), test_ts_ssim.mean(), test_ts_ssim.std(),
         test_st_mae.mean(), test_st_mae.std(), test_ts_mae.mean(), test_ts_mae.std())
  print(msg)
  print(msg_detail)

  if opts.result_dir:
    with open(os.path.join(opts.result_dir, "result.txt"), "w") as f:
      f.write(msg + msg_detail)

    numpy.save(os.path.join(opts.result_dir, "st_psnr.npy"), test_st_psnr)
    numpy.save(os.path.join(opts.result_dir, "ts_psnr.npy"), test_ts_psnr)
    numpy.save(os.path.join(opts.result_dir, "st_ssim.npy"), test_st_ssim)
    numpy.save(os.path.join(opts.result_dir, "ts_ssim.npy"), test_ts_ssim)
    numpy.save(os.path.join(opts.result_dir, "st_mae.npy"), test_st_mae)
    numpy.save(os.path.join(opts.result_dir, "ts_mae.npy"), test_ts_mae)


if __name__ == '__main__':
  main()
