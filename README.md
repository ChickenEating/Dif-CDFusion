# Bridging Spectral Fidelity and Structural Detail: A Diffusion-Based Common-Differential Fusion Network for Infrared-Visible Images

This repository provides the official PyTorch implementation of the paper ​**Bridging Spectral Fidelity and Structural Detail: A Diffusion-Based Common-Differential Fusion Network for Infrared-Visible Images**, which has been submitted to IEEE Transactions on Geoscience and Remote Sensing **(TGRS)**. In this work, we introduces ​Dif-CDFusion, a novel framework designed to address the critical challenge of reconciling spectral fidelity and structural consistency in infrared-visible image fusion. By leveraging diffusion-based feature extraction and a common-differential alternating fusion strategy, our approach achieves state-of-the-art performance in preserving both color integrity and structural details. This repository includes the complete source code, pretrained models, and evaluation scripts to facilitate reproducibility and further research in the field of multimodal image fusion. We hope this implementation will serve as a valuable resource for researchers and practitioners working on advanced image fusion techniques.

## Method Framework
Below is the framework of our proposed ​**Dif-CDFusion**:

![Dif-CDFusion Framework](./figs/framework.png)

## Abstract
Infrared and visible image fusion aims to enhance scene representation by integrating complementary sensor data. However, existing methods fail to reconcile spectral fidelity with structural consistency. For one thing, grayscale fusion approaches preserve structural details by discarding color information, inherently sacrificing spectral fidelity. For another, color fusion techniques maintain spectral authenticity but compromise details and structural consistency due to the misaligned chromatic information. To bridge the gap, we present the Dif-CDFusion, which resolves the conflict between spectral fidelity and the preservation of structural details through diffusion-based feature extraction and common-differential alternating feature fusion. By individually constructing a denoising diffusion process in latent space to model multi-channel spectral distributions, our approach extracts diffusion features that preserve color integrity while capturing complete spectral information for texture retention. Subsequently, we design a common-differential alternate fusion module to alternately integrate differential and common mode components within diffusion features, enhancing both structual details and thermal target salience. Extensive experiments demonstrate that our Dif-CDFusion achieves state-of-the-art performance both quantitatively and qualitatively.

## Results
Here are some qualitative results from our experiments:

### LLVIP Dataset
Visual comparisons of other state-of-the-art image fusion methods on image \#010434 from the LLVIP dataset:
![LLVIP-1](./figs/LLVIP-1.png)
Visual comparisons of other state-of-the-art image fusion methods on image \#130359 from the LLVIP dataset:
![LLVIP-2](./figs/LLVIP-2.png)

### MSRS Dataset
Visual comparisons of other state-of-the-art image fusion methods on image \#00633D from the MSRS dataset:
![MSRS-1](./figs/MSRS-1.png)
Visual comparisons of other state-of-the-art image fusion methods on image \#01356D from the MSRS dataset:
![MSRS-2](./figs/MSRS-2.png)

### VEDAI Dataset
Visual comparisons of other state-of-the-art image fusion methods on image \#00000027 from the VEDAI dataset:
![VEDAI-1](./figs/VEDAI-1.png)
Visual comparisons of other state-of-the-art image fusion methods on image \#00000396 from the VEDAI dataset:
![VEDAI-2](./figs/VEDAI-2.png)

## Usage
To train or test the model, follow these steps:
1. Download the datasets and pretrained weights.
   - ​**VEDAI**: [Download](https://downloads.greyc.fr/vedai/)
   - ​**LLVIP**: [Download](https://bupt-ai-cz.github.io/LLVIP/)
   - **MSRS**: [Download](https://github.com/Linfeng-Tang/MSRS)
   - **M3FD**: [Download](https://github.com/JinyuanLiu-CV/TarDAL)
   - ​**TNO**: [Download](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029)
   - **RoadScene**: [Download](https://github.com/hanna-xu/RoadScene)
   - ​**INO**: [Download](https://www.ino.ca/en/technologies/video-analytics-dataset/videos/)

   The pretrained weights for our model can be downloaded from the following Baidu netdisk link.

   链接: https://pan.baidu.com/s/1tEBDEbdg5PMKovgPf6j7YQ?pwd=1v2n 提取码: 1v2n

3. Clone this repository:
   ```bash
   git clone https://github.com/your_username/Dif-CDFusion.git
   cd Dif-CDFusion

4. Create a new conda environment with Python 3.8.20:
   ```bash
   conda create -n DifCDFusion python=3.8.20 --y
   conda activate DifCDFusion

5. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

6. Run the training or evaluation script:
   ```bash
   python train.py  # For training and evaluation
   python test.py   # For testing

## Citation
If you find this work useful, please cite our paper:
   ```bash
   @article{dif_cdfusion,
     title={Bridging Spectral Fidelity and Structural Detail: A Diffusion-Based Common-Differential Fusion Network for Infrared-Visible Images},
     author={Guanyu Liu, Ruiheng Zhang, Lixin Xu, Qi Zhang, and Daming Zhou},
     journal={IEEE Transactions on Geoscience and Remote Sensing},
     year={2025},
     note={Submitted}
   }
   ```
## Contact
For any questions or suggestions, please contact guanyu.liu@bit.edu.cn.

