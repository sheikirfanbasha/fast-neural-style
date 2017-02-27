echo "Start Time : `date +"%T"`" > file.txt
/usr/bin/time -v th slow_neural_style.lua -style_image images/styles/swirl.jpg -content_image images/content/fontli-with-clouds.png -content_weights 20 -style_weights 1000 -gpu 0 -output_image _experiments/fontli-cuda-adam-swirl-20-1000/out.png -backend cuda -optimizer adam  >> file.txt

sed -i "2i End time: `date +"%T"`" file.txt
mutt -s "Swirl-Style-With-Adam-Optimizer2" -a file.txt < /dev/null -- irfan.sheik@imaginea.com
