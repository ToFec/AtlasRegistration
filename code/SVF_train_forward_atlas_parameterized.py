from torch.utils.data import DataLoader
from torch import optim
import torch
import numpy as np
import datetime
import os
import sys
from atlasDataModule import AtlasDataModule
from atlasModule import AtlasModule
from imageTransformation import Bilinear
sys.path.append(os.path.realpath(".."))
import warnings
import argparse
import random
import torch.backends.cudnn as cudnn

from atlas_models import SVF_resid
from config import Config
from lossCalculator import LossCalculator

from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

def getOptimizer(optimizerType):
  if optimizerType == 'sgd':
    return partial(torch.optim.SGD, momentum=0.9, nesterov=True)
  return torch.optim.AdamW

def setSeeds(seed = 0):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  cudnn.deterministic = True
  pl.seed_everything(seed,workers=True)


parser = argparse.ArgumentParser(description='Atlas Registration')
parser.add_argument("-c", "--configFile", dest="configFile", help="configuration file")


if __name__ == "__main__":

    args = parser.parse_args()

    configFile = args.configFile
    if configFile:
      config = Config(configFile)
    else:
      config = Config()    
      
    seed = config.getParam("seed")

    if seed is not None:
      setSeeds(seed)

    gpu = config.getParam("gpu")
    if gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    max_epochs = config.getParam("epochs")
    save_per_epoch = config.getParam("saveEveryEpoch")
    batch_size = config.getParam("batchSize")
    lr = config.getParam("learningRate")
    atlas_lr = config.getParam("learningRate")
    loss_name = config.getParam("similarityLoss")
    best_score = 0.0

    reg_factor = config.getParam("regularizationFactor")
    sim_factor = config.getParam("similarityFactor")
    pair_sim_factor = config.getParam("imagePairSimFactor")
    smooth_factor = config.getParam("smoothingFactor")

    using_affine_init = config.getParam("affineInitialization")

    dataModule = AtlasDataModule(config)
    network = SVF_resid(img_sz=np.array([80, 192, 192]), args=args)
    lossCalculator = LossCalculator(config)
    
    loss = LossCalculator(config)
    optimizer = getOptimizer(config.getParam('optimizer'))
    
    data = AtlasDataModule(config)
    data.prepare_data()
    data.setup(stage="fit")
    
    initialAtlas = data.getInitalAtlas()
    
    model = AtlasModule(
        net=network,
        criterion=loss,
        learning_rate=config.getParam('learningRate'),
        optimizer_class=optimizer,
        useLrScheduler=config.getParam('lrScheduler'),
        initialAtlas
    )
        
    
    
    callBackFunctions=[]
      # early_stopping = pl.callbacks.early_stopping.EarlyStopping(
      #     monitor='val_loss',
      # )
      # callBackFunctions.append(early_stopping)
    stringForStoringVariables = "atlasRegistration" + + str(loss_name) \
                      + '_affine_init_' + str(int(using_affine_init)) \
                      + '_seed_' + str(seed) \
                      + '_reg_' + str(reg_factor) \
                      + '_atlas_sim_' + str(sim_factor) \
                      + '_pair_sim_' + str(pair_sim_factor) \
                      + '_smooth_' + str(smooth_factor) \
                      + '_epoch_' + str(max_epochs) \
                      + '_batchsize_' + str(batch_size) \
                      + '_network_lr_' + str(lr) \
                      + '_atlas_lr_' + str(atlas_lr)
                      
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./checkpoints/',
                                                       filename= stringForStoringVariables + '-{epoch:02d}-{val_loss:.2f}',
                                                       every_n_epochs=save_per_epoch,
                                                       monitor="val_loss",
                                                       mode="min",
                                                       save_top_k=3)
    callBackFunctions.append(checkpoint_callback)
    
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callBackFunctions.append(lr_monitor)  
    
    logger = TensorBoardLogger("tb_logs", name=stringForStoringVariables)
    
    trainer = pl.Trainer(
          gpus=1,
          precision=32,
          callbacks=callBackFunctions,
          auto_lr_find=config.getParam('tuneLR'),
          # profiler="simple",
          logger=logger,
          deterministic=True,
          check_val_every_n_epoch=5
      )
      
    trainer.tune(model,datamodule=data)
      
    trainer.logger._default_hp_metric = False
      
    start = datetime.now()
      
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)
    
    return
    

    # SVFNet_train = OAI_Atlas_Opt_3D(train_single_list)
    # SVFNet_train_dataloader = DataLoader(SVFNet_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # SVFNet_atlas_update = OAI_Atlas_Opt_3D(train_single_list)
    # SVFNet_atlas_update_dataloader = DataLoader(SVFNet_atlas_update, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    # SVFNet_val = OAI_Atlas_Opt_3D(valid_single_list)
    # SVFNet_val_dataloader = DataLoader(SVFNet_val, batch_size=1, shuffle=False, num_workers=config.getParam("numberOfWorkersDataLoader"), pin_memory=True)

    experiment_name = 'CVPR22_OAI_3D_SVFNet_Forward_Atlas_Parameterize_' + str(loss_name) \
                      + '_affine_init_' + str(int(using_affine_init)) \
                      + '_seed_' + str(seed) \
                      + '_reg_' + str(reg_factor) \
                      + '_atlas_sim_' + str(sim_factor) \
                      + '_pair_sim_' + str(pair_sim_factor) \
                      + '_smooth_' + str(smooth_factor) \
                      + '_epoch_' + str(max_epochs) \
                      + '_batchsize_' + str(batch_size) \
                      + '_network_lr_' + str(lr) \
                      + '_atlas_lr_' + str(atlas_lr)
                      
    
    train_model = SVF_resid(img_sz=np.array([80, 192, 192]), args=args)
    train_model.weights_init()
    if gpu is not None:
        torch.cuda.set_device(gpu)
        train_model.cuda(gpu)
    else:
        train_model.cuda()

    optimizer = optim.Adam(train_model.parameters(), lr=lr)

    train_model.train()
    now = datetime.datetime.now()
    now_date = "{:02d}{:02d}{:02d}".format(now.month, now.day, now.year)
    now_time = "{:02d}{:02d}{:02d}".format(now.hour, now.minute, now.second)
    writer = SummaryWriter(os.path.join('./logs', now_date, experiment_name + '_' + now_time))

    ## get initial alignment avg dice and initial atlas
    atlas_img = sitk.ReadImage('/playpen-raid/zpd/remote/Atlas/AtlasBuilding/initial_avg_atlas/averaged_initial_atlas_img.nii.gz')
    atlas_seg = torch.load('/playpen-raid/zpd/remote/Atlas/AtlasBuilding/initial_avg_atlas/averaged_initial_atlas_prob_seg.pt')
    atlas_img_array = sitk.GetArrayFromImage(atlas_img)
    atlas_tensor = torch.from_numpy(atlas_img_array).unsqueeze(0).unsqueeze(0)

    atlas_tensor.requires_grad = True
    atlas_optimizer = optim.SGD([atlas_tensor], lr=atlas_lr, weight_decay=0)

    bilinear = Bilinear(zero_boundary=False)

    img_sz = np.array([80, 192, 192])
    batch_sz = batch_size
    train_identity_map = gen_identity_map(img_sz).unsqueeze(0).repeat(batch_sz, 1, 1, 1, 1).cuda(gpu)
    val_identity_map = gen_identity_map(img_sz).unsqueeze(0).cuda(gpu)

    for epoch in range(max_epochs):
        atlas_optimizer.zero_grad()
        atlas_imgs = atlas_tensor.repeat(batch_sz, 1, 1, 1, 1).cuda(gpu)
        atlas_segs = atlas_seg.repeat(batch_sz, 1, 1, 1, 1).cuda(gpu)
        for i, (src_imgs, src_segs, src_ids) in enumerate(SVFNet_train_dataloader):
            global_step = epoch * len(SVFNet_train_dataloader) + (i + 1) * batch_size
            src_imgs, src_segs = src_imgs.cuda(gpu), src_segs.cuda(gpu)
            optimizer.zero_grad()

            cat_input = torch.cat((atlas_imgs, src_imgs), 1)
            pos_flow, neg_flow = train_model(cat_input)
            pos_deform_field = pos_flow + train_identity_map
            neg_deform_field = neg_flow + train_identity_map

            svf_warped_atlas_imgs = bilinear(atlas_imgs, pos_deform_field)
            svf_warped_atlas_segs = bilinear(atlas_segs, pos_deform_field)

            ## to evaluate in image space
            if pair_sim_factor != 0.0:
                sec_pos_deform_field = torch.flip(pos_deform_field, dims=[0])
                sec_src_imgs = torch.flip(src_imgs, dims=[0])
                sec_src_segs = torch.flip(src_segs, dims=[0])
                svf_warped_src_imgs_in_image_space = bilinear(src_imgs, (bilinear(neg_flow, sec_pos_deform_field) + sec_pos_deform_field))
                svf_warped_src_segs_in_image_space = bilinear(src_segs, (bilinear(neg_flow, sec_pos_deform_field) + sec_pos_deform_field))


            ## loss
            sim_loss = get_sim_loss(svf_warped_atlas_imgs, src_imgs, loss_name)
            reg_loss = get_reg_loss(pos_flow)
            if pair_sim_factor != 0.0:
                pair_sim_loss = get_pair_sim_loss_image_space(svf_warped_src_imgs_in_image_space, sec_src_imgs, loss_name)

            if pair_sim_factor == 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss
            elif pair_sim_factor != 0.0:
                loss = sim_factor * sim_loss + reg_factor * reg_loss + pair_sim_factor * pair_sim_loss

            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/training', loss.item(), global_step=global_step)
            if pair_sim_factor == 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}'.format(
                        epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item())
                )
            elif pair_sim_factor != 0.0:
                print('epoch {}, iter {}, total loss: {}, sim_factor: {}, sim_loss: {}, reg_factor: {}, reg_loss: {}, pair_sim_factor: {}, pair_sim_loss: {}'.format(
                    epoch, i + 1, loss.item(), sim_factor, sim_loss.item(), reg_factor, reg_loss.item(), pair_sim_factor, pair_sim_loss.item())
                )

            del svf_warped_atlas_imgs, svf_warped_atlas_segs, pos_flow, pos_deform_field, neg_flow, neg_deform_field
        with torch.no_grad():
            atlas_tensor.grad = atlas_tensor.grad/len(SVFNet_train_dataloader)
        smooth_loss = smooth_factor * get_first_order_reg_loss(atlas_tensor)
        smooth_loss.backward()
        atlas_optimizer.step()

        ## Validate to save the best atlas and model parameters
        if epoch % save_per_epoch == (save_per_epoch - 1):
            with torch.set_grad_enabled(False):
                ## create avg seg
                tmp_img, tmp_seg = 0, 0
                dice_all = 0
                atlas_imgs = atlas_tensor.cuda(gpu)
                atlas_segs = atlas_seg.cuda(gpu)
                for _, (mean_src_imgs, mean_src_segs, _) in enumerate(SVFNet_val_dataloader):
                    mean_src_imgs, mean_src_segs = mean_src_imgs.cuda(gpu), mean_src_segs.cuda(gpu)

                    src_cat_input = torch.cat((atlas_imgs, mean_src_imgs), 1)
                    _, mean_neg_flow_src = train_model(src_cat_input)
                    mean_neg_deform_field_src = mean_neg_flow_src + val_identity_map

                    mean_warped_src_segs = bilinear(mean_src_segs, mean_neg_deform_field_src)

                    tmp_seg += mean_warped_src_segs
                mean_atlas_seg_tensor = tmp_seg / len(SVFNet_val_dataloader)

                ## inference
                for _, (inf_src_imgs, inf_src_segs, _) in enumerate(SVFNet_val_dataloader):
                    inf_src_imgs, inf_src_segs = inf_src_imgs.cuda(gpu), inf_src_segs.cuda(gpu)
                    src_cat_input = torch.cat((atlas_imgs, inf_src_imgs), 1)
                    pos_flow, _ = train_model(src_cat_input)

                    pos_deform_field = pos_flow + val_identity_map
                    svf_warped_atlas_segs = bilinear(mean_atlas_seg_tensor, pos_deform_field)

                    dice_all += (1.0 - get_atlas_seg_loss(inf_src_segs, svf_warped_atlas_segs))

                dice_avg = dice_all / len(SVFNet_val_dataloader)
                print("{} epoch, {} iter, training loss: {:.5f}, val dice: {:.5f}".format(epoch, i + 1, loss.item(), dice_avg))
                writer.add_scalar('validation/dice_avg', dice_avg, global_step=global_step)


                if dice_avg > best_score:
                    best_score = dice_avg.item()
                    print('{} epoch, current highest - Dice: {:.5f}'.format(epoch, dice_avg))
                    writer.add_scalar('validation/highest_dice', dice_avg, global_step=global_step)
                    save_model_path = './ckpoints/' + experiment_name + '/'
                    if not os.path.isdir(save_model_path):
                        os.mkdir(save_model_path)
                    best_state = {'epoch': epoch,
                                  'state_dict': train_model.state_dict(),
                                  'optimizer': optimizer.state_dict(),
                                  'best_score': best_score,
                                  'global_step': global_step
                                  }
                    torch.save(best_state, save_model_path + 'model_best.pth.tar')
                    tmp_img, tmp_seg, JD_denominator = 0, 0, 0
                    for _, (update_src_imgs, update_src_segs, _) in enumerate(SVFNet_atlas_update_dataloader):
                        update_src_imgs, update_src_segs = update_src_imgs.cuda(gpu), update_src_segs.cuda(gpu)

                        src_cat_input = torch.cat((atlas_imgs, update_src_imgs), 1)
                        update_pos_flow_src, update_neg_flow_src = train_model(src_cat_input)
                        update_pos_deform_field_src = update_pos_flow_src + val_identity_map
                        update_neg_deform_field_src = update_neg_flow_src + val_identity_map

                        update_warped_src_segs = bilinear(update_src_segs, update_neg_deform_field_src)

                        JD_tensor = torch.from_numpy(jacobian_determinant(update_neg_deform_field_src)).unsqueeze(0).unsqueeze(0).cuda(gpu)
                        JD_denominator += JD_tensor

                        tmp_seg += (update_warped_src_segs*JD_tensor)
                    atlas_seg = tmp_seg / JD_denominator
                    save_atlas_img_name = save_model_path + 'atlas_svf_img_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.nii.gz'
                    save_atlas_est_name = save_model_path + 'atlas_svf_est_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.nii.gz'
                    save_atlas_prob_name = save_model_path + 'atlas_svf_prob_epoch_' + str(1000+epoch) + '_' + loss_name + '_' + str(best_score) + '.pt'
                    save_updated_atlas(atlas_tensor, atlas_seg, save_atlas_img_name, save_atlas_est_name, save_atlas_prob_name)

            save_model_path = './ckpoints/' + experiment_name + '/'
            if not os.path.isdir(save_model_path):
                os.mkdir(save_model_path)
            current_state = {'epoch': epoch,
                             'state_dict': train_model.state_dict(),
                             'optimizer': optimizer.state_dict(),
                             'best_score': best_score,
                             'global_step': global_step
                            }
            torch.save(current_state, save_model_path + 'checkpoint.pth.tar')



    writer.close()