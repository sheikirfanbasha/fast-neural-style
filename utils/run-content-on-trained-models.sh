#This script will be useful to run the content images placed in a directory over the pretrained 
# models of different styles which are checkpointed for every 2000 iterations.
# The resulting output will be saved to a folder with the content name
styles_dir="/Users/irfan/Imaginea/Experiments/fast-neural-style/images/styles/custom_styles"/*
content_dir="/Users/irfan/Imaginea/Experiments/fast-neural-style/images/content/custom_content"/*
limit=20
for contentName in $content_dir
do
	echo "contentName :"$contentName >> file.txt
	IFS='.' read -a arr <<< "$contentName"
	folderName="$(basename $arr)"
	mkdir -p $folderName
	for styleName in $styles_dir
	do
	  echo "styleName :"$styleName >> file.txt
	  IFS='.' read -a myarray <<< "$styleName"
	  for i in $(seq 1 $limit); do
		   	modelSequence=$(($i * 2000))
		   	underscore="_"
		   	slash="/"
		   	styleratio="1_200$underscore"
	  		checkpoint_name="$(basename $myarray)$styleratio$modelSequence.t7"
	  		output_image_name="$folderName$slash$(basename $myarray)$underscore$styleratio$modelSequence.png"
	  		echo "model name: $checkpoint_name" >> file.txt
	  		echo "output name: $output_image_name" >> file.txt
	  		#th fast_neural_style.lua   -model $checkpoint_name   -input_image $contentName   -output_image $output_image_name   -gpu 0   -backend cuda    -use_cudnn 1 >> file.txt
		done
	  
	done
done

echo "finished for all the styles" >> file.txt
