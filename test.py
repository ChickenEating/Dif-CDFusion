import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torchvision.transforms.functional import rgb_to_grayscale
import argparse, time
import models as Model
import options.options as option
import numpy as np
from utils import util
from utils.evaluator import Evaluator
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
from PIL import Image
import warnings

# 忽略 UserWarning 类型的警告
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    #    opt = option.parse(parser.parse_args().opt)
    opt = option.parse('./options/test/test_ATFuse.json')
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names = []
    test_loaders = []
    for testset_name, dataset_opt in opt['datasets'].items():
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print(f'===> Test Dataset: [{testset_name}]   Number of images: [{len(test_set)}]')
        bm_names.append(testset_name)
    
    # create solver (and load model)
    solver = create_solver(opt)  ### load train and test model
    
    # 初始化diffusion特征提取网络
    with torch.no_grad():
        diffusion_ir = Model.create_model(opt, channel=3)
        diffusion_vis = Model.create_model(opt, channel=3)
    
        # Set noise schedule for the diffusion model
        diffusion_ir.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        diffusion_vis.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    print('===> Start Test')
    print("==================================================")
    print(f"Method: {model_name} || Scale: {scale} || Degradation: {degrad}")

    diffusion_ir.netG.eval()
    diffusion_vis.netG.eval()

    for bm, test_loader in zip(bm_names, test_loaders):
        if not bm == "LLVIP":
            continue
        
        print(f"Test set : [{bm}]")
        total_time = []
        metric_result = np.zeros((10))
        with torch.no_grad():
            for iter, batch in enumerate(test_loader):
                t0 = time.time()
                diffusion_ir.feed_data(batch['Ir'])
                diffusion_vis.feed_data(batch['Vi'])
                for t in opt['model_df']['t']:
                    fe_t_ir, fd_t_ir = diffusion_ir.get_feats(t=t)
                    fe_t_vi, fd_t_vi = diffusion_vis.get_feats(t=t)
                    if opt['model_df']['feat_type'] == "dec":
                        feature = {'ViF': fd_t_ir, 'IrF': fd_t_vi}
                        del fe_t_ir, fd_t_ir, fe_t_vi, fd_t_vi
                    else:
                        feature = {'ViF': fe_t_ir, 'IrF': fe_t_vi}
                        del fe_t_ir, fd_t_ir, fe_t_vi, fd_t_vi
                solver.feed_data(feature, batch)
                del feature
                solver.test(opt)
                t1 = time.time()
                total_time.append((t1 - t0))
                visuals = solver.get_current_visual()
                
                torch.cuda.empty_cache()
                
                save_img_path = os.path.join('./test_results/', bm)
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
                solver.save_K_V(visuals, save_img_path, os.path.basename(batch["vi_path"][0])) # 保存注意力特征图
                save_img_path = os.path.join(save_img_path, os.path.basename(batch["vi_path"][0]))
                Fu = visuals['Fu'].numpy().transpose(1, 2, 0)
                # Fu = cv2.cvtColor(Fu, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, Fu)
                print(f"===> Saving Fu images of [{bm}]... Save Path: [{save_img_path}]")
                print(f"Saving Fu images of [{bm}], time: {t1 - t0}")
                # 灰度化并转换为 numpy 数组（如果需要）
                Fusion = rgb_to_grayscale(visuals['Fu']).squeeze(0).numpy()  # visuals['Fu'] -> 灰度图
                IR = rgb_to_grayscale(batch['Ir'].squeeze(0)).squeeze(0).numpy()  # batch['Ir'] -> 灰度图
                VI = rgb_to_grayscale(batch['Vi'].squeeze(0)).squeeze(0).numpy()  # batch['Vi'] -> 灰度图

                CC, RMSE = util.calc_metrics(visuals['Fu'], batch['Ir'].squeeze(0), batch['Vi'].squeeze(0)) # 计算CC, RMSE
                # 修改后的指标计算代码，使用灰度化后的输入
                metric_result += np.array([
                    CC, RMSE, 
                    Evaluator.EN(Fusion),              # 计算熵
                    Evaluator.SD(Fusion),              # 计算标准差
                    Evaluator.SF(Fusion),              # 计算空间频率
                    Evaluator.MI(Fusion, IR, VI),      # 计算互信息
                    Evaluator.SCD(Fusion, IR, VI),     # 计算结构相似度差异
                    Evaluator.VIFF(Fusion, IR, VI),    # 计算 VIFF 指标
                    Evaluator.Qabf(Fusion, IR, VI),    # 计算边缘保持性
                    Evaluator.SSIM(Fusion, IR, VI)     # 计算结构相似性
                ])
            metric_result /= len(os.listdir(test_loader))
            
            print("\t\t EN\t\t SD\t\t SF\t\t MI\t\t SCD\t\t VIF\t\t Qabf\t\t SSIM")
            print(
                "\t\t "
                + str(np.round(metric_result[0], 4))
                + "\t\t"
                + str(np.round(metric_result[1], 4))
                + "\t\t"
                + str(np.round(metric_result[2], 4))
                + "\t\t"
                + str(np.round(metric_result[3], 4))
                + "\t\t"
                + str(np.round(metric_result[4], 4))
                + "\t\t"
                + str(np.round(metric_result[5], 4))
                + "\t\t"
                + str(np.round(metric_result[6], 4))
                + "\t\t"
                + str(np.round(metric_result[7], 4))
            )
            print(f"{bm} Dataset Evaluation Finished !")
            print(f"---- Average time for [{bm}] is {sum(total_time) / len(total_time)} sec ----")
    print("==================================================")
    print("===> All datasets Finished !")


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    main()
