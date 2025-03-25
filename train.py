# -*- coding: utf-8 -*-
import argparse, random
from tqdm import tqdm
import torch
import models as Model
from visualization import get_local
import cv2
import torch.nn as nn
torch.cuda.empty_cache()
get_local.activate()

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def parse_options(option_file_path):
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(option_file_path)
    return opt


def set_random_seed(opt):
    seed = opt['solver']['manual_seed']
    if seed is None: seed = random.randint(1, 10000)  # 这里尽量直接把seed写死
    print(f"===> Random Seed: [{seed}]")
    random.seed(seed)
    torch.manual_seed(seed)


def create_Dataloader(opt):
    train_set, train_loader, val_set, val_loader = None, None, None, None
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print(f'===> Train Dataset: {train_set.name()}   Number of images: [{len(train_set)}]')
            if train_loader is None: raise ValueError("[Error] The training data does not exist")
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print(f'===> Val Dataset: {val_set.name()}   Number of images: [{len(val_set)}]')
        else:
            raise NotImplementedError(f"[Error] Dataset phase [{phase}] in *.json is not recognized.")
    return train_set, train_loader, val_set, val_loader

import torch
import torch.nn.functional as F


def main():
    option_file_path = "./options/train/train_ATFuse.json"
    opt = parse_options(option_file_path)
    # random seed
    set_random_seed(opt)

    # create train and val dataloader
    train_set, train_loader, val_set, val_loader = create_Dataloader(opt)

    # 初始化网络
    solver = create_solver(opt)

    scale = opt['scale']
    model_name = opt['networks']['which_model'].upper()
    solver_log = solver.get_current_log()

    NUM_EPOCH = int(opt['solver']['num_epochs'])
    start_epoch = solver_log['epoch']

    # 初始化diffusion特征提取网络
    with torch.no_grad():
        diffusion_ir = Model.create_model(opt, channel=3)
        diffusion_vis = Model.create_model(opt, channel=3)
    
        # Set noise schedule for the diffusion model
        diffusion_ir.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        diffusion_vis.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    print('===> Start Train')
    print("==================================================")
    print(f"Method: {model_name} || Scale: {scale} || Epoch Range: ({start_epoch} ~ {NUM_EPOCH})")
    solver_log['best_pred'] = 10000000000
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        print(f'\n===> Training Epoch: [{epoch}/{NUM_EPOCH}]...  Learning Rate: {solver.get_current_learning_rate():.2e}')

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        get_local.clear()

        train_loss_list = []
        diffusion_ir.netG.eval()
        diffusion_vis.netG.eval()
        with tqdm(total=len(train_loader), desc=f'Epoch: [{epoch}/{NUM_EPOCH}]', miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                with torch.no_grad():
                    diffusion_ir.feed_data(batch['Ir'])
                    diffusion_vis.feed_data(batch['Vi'])
                    for time in opt['model_df']['t']:
                        fe_t_ir, fd_t_ir = diffusion_ir.get_feats(t=time)
                        fe_t_vi, fd_t_vi = diffusion_vis.get_feats(t=time)
                        if opt['model_df']['feat_type'] == "dec":
                            # feature = {'ViF': batch['Vi'], 'IrF': batch['Ir']}
                            feature = {'ViF': fd_t_ir, 'IrF': fd_t_vi}
                            del fe_t_ir, fd_t_ir, fe_t_vi, fd_t_vi
                        else:
                            # feature = {'ViF': batch['Vi'], 'IrF': batch['Ir']}
                            feature = {'ViF': fe_t_ir, 'IrF': fe_t_vi}
                            del fe_t_ir, fd_t_ir, fe_t_vi, fd_t_vi
                solver.feed_data(feature, batch)
                del feature
                iter_loss = solver.train_step(opt)
                batch_size = batch['Vi'].size(0)
                train_loss_list.append(iter_loss * batch_size)
                t.set_postfix_str("Batch Loss: %.4f" % iter_loss)
                t.update()
                # cache = get_local.cache
                # print(cache)

        solver_log['records']['train_loss'].append(sum(train_loss_list) / len(train_set))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print(f'\nEpoch: [{epoch}/{NUM_EPOCH}]   Avg Train Loss: {sum(train_loss_list) / len(train_set):.6f}')

        print('===> Validating...', )

        cc_list = []
        rmse_list = []
        val_loss_list = []
        diffusion_ir.netG.eval()
        diffusion_vis.netG.eval()
        solver.model.eval()
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(val_loader)):
                diffusion_ir.feed_data(batch['Ir'])
                diffusion_vis.feed_data(batch['Vi'])
                for tim in opt['model_df']['t']:
                    fe_t_ir, fd_t_ir = diffusion_ir.get_feats(t=tim)
                    fe_t_vi, fd_t_vi = diffusion_vis.get_feats(t=tim)
                    if opt['model_df']['feat_type'] == "dec":
                        feature = {'ViF': batch['Vi'], 'IrF': batch['Ir']}
                        feature = {'ViF': fd_t_ir, 'IrF': fd_t_vi}
                        del fe_t_ir, fd_t_ir, fe_t_vi, fd_t_vi
                    else:
                        feature = {'ViF': batch['Vi'], 'IrF': batch['Ir']}
                        feature = {'ViF': fe_t_ir, 'IrF': fe_t_vi}
                        del fe_t_ir, fd_t_ir, fe_t_vi, fd_t_vi
                solver.feed_data(feature, batch)
                del feature
                iter_loss = solver.test(opt)
                val_loss_list.append(iter_loss)
                del iter_loss
                
                torch.cuda.empty_cache()
                
                if opt["save_image"] and epoch % 1 == 0:
                    solver.save_current_visual(epoch, iter, batch['Ir'].squeeze(0), batch['Vi'].squeeze(0))
                
            # calculate evaluation metrics
            visuals = solver.get_current_visual()
            
            save_img_path = './feature_maps/'
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            solver.save_K_V(visuals, save_img_path, os.path.basename(batch["vi_path"][0])) # 保存注意力特征图
            
            cc, rmse = util.calc_metrics(visuals['Fu'], batch['Vi'], batch['Ir'])
            cc_list.append(cc)
            rmse_list.append(rmse)
        
        solver_log['records']['val_loss'].append(sum(val_loss_list) / len(val_loss_list))
        solver_log['records']['psnr'].append(sum(cc_list) / len(cc_list))
        solver_log['records']['ssim'].append(sum(rmse_list) / len(rmse_list))

        epoch_is_best = False
        if solver_log['best_pred'] > (sum(val_loss_list) / len(val_loss_list)):
            solver_log['best_pred'] = (sum(val_loss_list) / len(val_loss_list))
            epoch_is_best = True
            solver_log['best_epoch'] = epoch

        print(
            f"[{val_set.name()}] CC: {sum(cc_list) / len(cc_list):.4f}   RMSE: {sum(rmse_list) / len(rmse_list):.4f} "
            f"Loss: {sum(val_loss_list) / len(val_loss_list):.6f}   Best : {solver_log['best_pred']:.6f} in Epoch: "
            f"[{solver_log['best_epoch']:d}]"
        )
        
        del cc_list, rmse_list, val_loss_list
        
        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, epoch_is_best)
        solver.save_current_log()
        solver.update_learning_rate()
        torch.cuda.empty_cache()

    print('===> Finished !')


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    main()
