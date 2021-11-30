import torch
from options_pelvic import TestOptions
from dataset import dataset_single
from model import DRIT
from saver import save_imgs
import os
import sys
import numpy
import pdb
import skimage.io

sys.path.append(os.path.join("..", "..", "util"))
import common_metrics
import common_pelvic_pt as common_pelvic

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

  test_ids_t = common_pelvic.load_data_ids(opts.dataroot, "testing", "treat")
  test_data_s, test_data_t, _, _ = common_pelvic.load_test_data(opts.dataroot, mini=opts.mini)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(0)
  model.resume(os.path.join(opts.checkpoint_dir, "final.pth"), train=False)
  model.eval()

  test_st_psnr = numpy.zeros((test_data_s.shape[0], 1), numpy.float32)
  test_ts_psnr = numpy.zeros((test_data_t.shape[0], 1), numpy.float32)
  test_st_list = []
  test_ts_list = []
  msg_detail = ""
  with torch.no_grad():
    for i in range(test_data_s.shape[0]):
      test_st = numpy.zeros(test_data_s.shape[1:], numpy.float32)
      test_ts = numpy.zeros(test_data_t.shape[1:], numpy.float32)
      used = numpy.zeros(test_data_s.shape[1:], numpy.float32)
      for j in range(test_data_s.shape[1] - opts.input_dim_a + 1):
        test_patch_s = torch.tensor(test_data_s[i:i + 1, j:j + opts.input_dim_a, :, :], device=device)
        test_patch_t = torch.tensor(test_data_t[i:i + 1, j:j + opts.input_dim_a, :, :], device=device)

        ret_st = model.test_forward_transfer(test_patch_s, test_patch_t, a2b=True)
        ret_ts = model.test_forward_transfer(test_patch_t, test_patch_s, a2b=False)

        test_st[j:j + opts.input_dim_a, :, :] += ret_st.cpu().detach().numpy()[0]
        test_ts[j:j + opts.input_dim_a, :, :] += ret_ts.cpu().detach().numpy()[0]
        used[j:j + opts.input_dim_a, :, :] += 1

      assert used.min() > 0
      test_st /= used
      test_ts /= used

      if opts.result_dir:
        common_pelvic.save_nii(test_ts, os.path.join(opts.result_dir, "syn_%s.nii.gz" % test_ids_t[i]))

      st_psnr = common_metrics.psnr(test_st, test_data_t[i])
      ts_psnr = common_metrics.psnr(test_ts, test_data_s[i])

      test_st_psnr[i] = st_psnr
      test_ts_psnr[i] = ts_psnr
      test_st_list.append(test_st)
      test_ts_list.append(test_ts)

      msg_detail += "  %s_psnr: %f\n" % (test_ids_t[i], ts_psnr)

  msg = "  test_st_psnr:%f/%f  test_ts_psnr:%f/%f" % \
        (test_st_psnr.mean(), test_st_psnr.std(), test_ts_psnr.mean(), test_ts_psnr.std())
  print(msg)
  print(msg_detail)

  if opts.result_dir:
    with open(os.path.join(opts.result_dir, "result.txt"), "w") as f:
      f.write(msg + msg_detail)

    numpy.save(os.path.join(opts.result_dir, "st_psnr.npy"), test_st_psnr)
    numpy.save(os.path.join(opts.result_dir, "ts_psnr.npy"), test_ts_psnr)


if __name__ == '__main__':
  main()
