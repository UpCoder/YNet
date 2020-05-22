### TF_SLIM_Framework
- Run the XX_to_tfrecords generate tfrecords file
- Write the code about XX_get_dataset to resolve the tfrecord file
- Run train_segmentation.py (default is UNet based on VGG16)
- Run test_segmentation.py


## How to Run
# training
```bash
python -u /home/give/PycharmProjects/weakly_label_segmentation/train_segmentation_ISBI2017_V2.py --train_dir=/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly_V3 --num_gpus=1 --learning_rate=1e-4 --gpu_memory_fraction=1 --train_image_width=256 --train_image_height=256 --batch_size=2 --dataset_dir=/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_2/tfrecords_V2_V2 --dataset_name=isbi2017v2 --dataset_split_name=train --max_number_of_steps=10000000 --checkpoint_path=/home/give/github/pixel_link/checkpoints/conv2_2/model.ckpt-73018 --using_moving_average=1 --decoder=upsampling --update_center_strategy=1 --update_center=True --attention_flag=False --num_centers_k=4 --full_annotation_flag=False --dense_connection_flag=False --learnable_connection_flag=False
```
    
- 参数解释
    - train_dir：string, 保存ck和events的路径
    - num_gpus: int, GPU的个数，目前只支持单个GPU
    - learning_rate： float
    - gpu_memory_fraction：float
    - train_image_width：int
    - train_image_height：int
    - batch_size：int
    - dataset_dir：string，tfrecords的文件
    - dataset_name：string，
    - dataset_split_name：string
    - max_number_of_steps：int，最大执行的步数
    - checkpoint_path：restore的路径
    - using_moving_average：
    - decoder:string, upsampling or transpose
    - update_center_strategy: int, only support 1
    - update_center: bool
    - attention_flag: bool
    - num_centers_k: int, 当更新策略等于1的时候无用
    - full_annotation_flag: bool，如果为True，则使用全监督，否则若监督
    - dense_connection_flag：bool
    - learnable_connection_flag：bool
# evaluation
```bash
python -u /home/give/PycharmProjects/weakly_label_segmentation/evulate_segmentation_ISBI2017_V2.py --checkpoint_path=/home/give/PycharmProjects/weakly_label_segmentation/logs/ISBI2017_V2/1s_agumentation_weakly-upsampling-2/model.ckpt-168090 --dataset_dir=/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1 --pred_vis_dir=/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/pred_vis --pred_dir=/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/pred --recovery_img_dir=/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/recovery_img_step --recovery_feature_map_dir=/home/give/Documents/dataset/ISBI2017/weakly_label_segmentation_V4/Batch_1/DLSC_0/recovery_feature_map --decoder=upsampling --using_moving_average=1 --batch_size=2 --update_center=True --test_flag=False --num_centers_k=4 --nii_flag=False --full_annotation_flag=False
```
- 参数解释
    - checkpoint_path: string，训练好的模型路径
    - dataset_dir：string,数据路径。暂时无用，在代码中硬指定
    - pred_vis_dir：string，暂时无用，在代码中硬指定
    - pred_dir：string，暂时无用，在代码中硬指定
    - recovery_img_dir：string，暂时无用，在代码中硬指定
    - recovery_feature_map_dir：string，暂时无用，在代码中硬指定
    - using_moving_average：string，无用
    - batch_size：int，无用
    - update_center：无用
    - test_flag：无用
    - num_centers_k：无用
    - nii_flag：无用
    - full_annotation_flag： bool