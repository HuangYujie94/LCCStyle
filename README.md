# LCCStyle

This is the offical implementation of the paper: "Y. Huang, M. Jing, J. Zhou, Y. Liu and Y. Fan, "LCCStyle: Arbitrary Style Transfer with Low Computational Complexity," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2021.3128058."

The paper can be found at: https://ieeexplore.ieee.org/document/9615003

Tensorflow 1.08-1.14



Train LCCStyleFC example:

python train_LCCStyleFC.py -is_training=true -vgg_model19='../../VGG19/vgg_19.ckpt' -train_content_path='../../Style_transfer/content/' -train_style_path='../../Style_transfer/style/' -style_w=7



Test LCCStyleFC example:

python train_LCCStyleFC.py -transfer_model='model_LCCStyleFC/genera20.0.ckpt' -test_data_path='content.jpg' -new_img_name='transfer.jpg' -test_style_path='style.jpg'

please cite the paper as:

"Y. Huang, M. Jing, J. Zhou, Y. Liu and Y. Fan, "LCCStyle: Arbitrary Style Transfer with Low Computational Complexity," in IEEE Transactions on Multimedia, doi: 10.1109/TMM.2021.3128058."

@ARTICLE{9615003,
  author={Huang, Yujie and Jing, Minge and Zhou, Jinjia and Liu, Yuhao and Fan, Yibo},
  journal={IEEE Transactions on Multimedia}, 
  title={LCCStyle: Arbitrary Style Transfer with Low Computational Complexity}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3128058}}
