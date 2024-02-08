import torch
from options_ixi import TrainOptions
from model import DRIT
from saver import Saver
import sys
import os
import numpy
import pdb
import skimage.io
import platform
from datetime import datetime


if platform.system() == 'Windows':
  sys.path.append(r"E:\我的坚果云\sourcecode\python\util")
else:
  sys.path.append("/home/chenxu/我的坚果云/sourcecode/python/util")
import common_metrics
import common_ixi

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  if opts.gpu >= 0:
      os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu)
      device = torch.device("cuda")
  else:
      os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
      device = torch.device("cpu")

  if not os.path.exists(opts.checkpoint_dir):
    os.makedirs(opts.checkpoint_dir)

  # daita loader
  print('\n--- load dataset ---')
  #torch.autograd.set_detect_anomaly(True)

  dataset_s = common_ixi.Dataset(opts.dataroot, modality="t2", n_slices=opts.input_dim_a, debug=opts.debug)
  dataset_t = common_ixi.Dataset(opts.dataroot, modality="t1", n_slices=opts.input_dim_a, debug=opts.debug)
  dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=opts.batch_size, shuffle=True, pin_memory=True, drop_last=True)
  dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=opts.batch_size, shuffle=True, pin_memory=True, drop_last=True)
  if opts.do_validation:
    val_data_t, val_data_s = common_ixi.load_test_data(opts.dataroot, "val")

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(0)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)

  # train
  print('\n--- train ---')
  best_psnr = 0
  for it in range(opts.max_epochs):
    for batch_id, (data_s, data_t) in enumerate(zip(dataloader_s, dataloader_t)):
      images_a = data_s["image"].to(device)
      images_b = data_t["image"].to(device)

      # update model
      if (batch_id + 1) % opts.d_iter != 0 and batch_id < len(dataloader_s) - 2:
        model.update_D_content(images_a, images_b)
        continue
      else:
        model.update_D(images_a, images_b)
        model.update_EG()

    if (it + 1) % opts.display_freq == 0 and opts.do_validation:
      val_st_psnr = numpy.zeros((val_data_s.shape[0], 1), numpy.float32)
      val_ts_psnr = numpy.zeros((val_data_t.shape[0], 1), numpy.float32)
      val_st_list = []
      val_ts_list = []
      with torch.no_grad():
        for i in range(val_data_s.shape[0]):
          val_st = numpy.zeros(val_data_s.shape[1:], numpy.float32)
          val_ts = numpy.zeros(val_data_t.shape[1:], numpy.float32)
          used = numpy.zeros(val_data_s.shape[1:], numpy.float32)
          for j in range(val_data_s.shape[1] - opts.input_dim_a + 1):
            val_patch_s = torch.tensor(val_data_s[i:i + 1, j:j + opts.input_dim_a, :, :], device=device)
            val_patch_t = torch.tensor(val_data_t[i:i + 1, j:j + opts.input_dim_a, :, :], device=device)

            ret_st = model.test_forward_transfer(val_patch_s, val_patch_t, a2b=True)
            ret_ts = model.test_forward_transfer(val_patch_t, val_patch_s, a2b=False)

            val_st[j:j + opts.input_dim_a, :, :] += ret_st.cpu().detach().numpy()[0]
            val_ts[j:j + opts.input_dim_a, :, :] += ret_ts.cpu().detach().numpy()[0]
            used[j:j + opts.input_dim_a, :, :] += 1

          assert used.min() > 0
          val_st /= used
          val_ts /= used

          st_psnr = common_metrics.psnr(val_st, val_data_t[i])
          ts_psnr = common_metrics.psnr(val_ts, val_data_s[i])

          val_st_psnr[i] = st_psnr
          val_ts_psnr[i] = ts_psnr
          val_st_list.append(val_st)
          val_ts_list.append(val_ts)

      msg = "%s Iter:%d  val_st_psnr:%f/%f  val_ts_psnr:%f/%f" % \
             (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), it + 1, val_st_psnr.mean(), val_st_psnr.std(), val_ts_psnr.mean(), val_ts_psnr.std())
      gen_images_test = numpy.concatenate([val_data_s[0], val_st_list[0], val_ts_list[0], val_data_t[0]], 2)
      gen_images_test = numpy.expand_dims(gen_images_test, 0).astype(numpy.float32)
      gen_images_test = common_ixi.generate_display_image(gen_images_test, is_seg=False)

      if opts.display_dir:
        try:
          skimage.io.imsave(os.path.join(opts.display_dir, "gen_images_test.jpg"), gen_images_test)
        except:
          pass

      if val_ts_psnr.mean() > best_psnr:
        best_psnr = val_ts_psnr.mean()

        if best_psnr > opts.psnr_threshold:
          model.save(os.path.join(opts.checkpoint_dir, "best.pth"), it, opts.max_epochs)

      msg += "  best_ts_psnr:%f" % best_psnr
      print(msg)

  model.save(os.path.join(opts.checkpoint_dir, "final.pth"), it, opts.max_epochs)


if __name__ == '__main__':
  main()
