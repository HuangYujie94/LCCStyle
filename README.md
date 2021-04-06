# LCCStyle

Tensorflow 1.08-1.14



Train LCCStyleFC example:
python train_LCCStyleFC.py -is_training=true -vgg_model19='../../VGG19/vgg_19.ckpt' -train_content_path='../../Style_transfer/content/' -train_style_path='../../Style_transfer/style/' -style_w=7



Test LCCStyleFC example:
python train_LCCStyleFC.py -transfer_model='model_LCCStyleFC/genera20.0.ckpt' -test_data_path='content.jpg' -new_img_name='transfer.jpg' -test_style_path='style.jpg'
