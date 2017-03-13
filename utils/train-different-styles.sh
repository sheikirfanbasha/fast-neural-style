#working_dir="/home/lab/irfan_experiments/fast-neural-style/images/styles/custom_styles"/*
working_dir="/Users/irfan/Imaginea/Experiments/fast-neural-style/images/styles/custom_styles"/*
for styleName in $working_dir
do
  echo "styleName :"$styleName >> file.txt
  IFS='.' read -a myarray <<< "$styleName"
  styleratio="1_200"
  checkpoint_name="$(basename $myarray)$styleratio"
  echo "name: $checkpoint_name"
  th train2.lua -h5_file _datamodels/ms-coco-2014.h5 -style_image $styleName -style_image_size 300 -content_weights 1.0 -style_weights 200.0 -checkpoint_name $checkpoint_name -checkpoint_every 1 -gpu 0 -backend cuda -use_cudnn 1 -display_every 2 -num_iterations 2 >> file.txt
done

echo "finished for all the styles" >> file.txt
