# RNN-plant-disease
This is the implementation of our [Frontiers in Plant Science 2020](https://www.frontiersin.org/articles/10.3389/fpls.2020.601250/full) work with titled -- Attention-Based Recurrent Neural Network for Plant Disease Classification. We study a new technique based on a recurrent neural network (RNN) to automatically locate infected regions and extract characteristics relevant for disease identification.
The project is open source under BSD-3 license (see the ``` LICENSE ``` file). Codes can be used freely only for academic purpose.

## Dependency
The codes are based on [Tensorflow](https://www.tensorflow.org/)

## Dataset
* The [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) used has has 38 crop-disease pairs, with 26 types of diseases for 14 crop plants.
* In order to train the purely disease classifier with the model using vernacular disease names, the leaf samples of PV dataset are first categorized into 21 classes (20 diseases and one healthy class).
These 21 classes are listed in  ```target_class.txt```


## Installation and Running

1. Users are required to install [tensorflow](https://www.tensorflow.org/) Library.

2. Users are required to download the necessary files [here](https://github.com/cs-chan/Deep-Plant/tree/master/PlantStructNet/Dataset) and the aforementioned dataset.

3. Users can first train a CNN model by setting the parameters as follows:
  * In RNN_multiOb_12_5.py (line 341): cross_entropy =  0 * self.cross_entropy_di + self.cross_entropy_cnn 
  * In main.py (line 342 - 345 ): Use 'model.optimize_ZOC', 'model.optimize_AOC', 'model.optimize_TSC'.

4. Next, to train the RNN model, freeze the CNN model by setting the parameters as follows:
* In RNN_multiOb_12_5.py (line 341): cross_entropy =  self.cross_entropy_di +  0 * self.cross_entropy_cnn  
* In main.py (line 342 - 345 ): Use 'model.optimize_ZOR', 'model.optimize_AOR', 'model.optimize_TSR'

Note that users are expected to modify the corresponding files to correct path to work properly. Enjoy!

## Dataset
* PlantClef image dataset can be downloaded [here](http://www.imageclef.org/lifeclef/2015/plant).

* Extract the convolutional features of the HGO-CNN pretrained using [caffe](http://caffe.berkeleyvision.org/). Details of the HGO-CNN can be obtained [here](https://ieeexplore.ieee.org/document/8359391/). 

	* After downloaded the PlantClef2015 dataset, users have to categorise the images into their respective species classes based on the information provided [here](http://www.imageclef.org/lifeclef/2015/plant)
 * Then, please run the ``` getfeatures.py ``` in the 'CNN' folder to extract the convolutional features from the pretrained [HGO-CNN](PlantClef_vgg_species_organ_iter_180854.caffemodel) model.
	    Mean file ``` species_mean_aug.npy ``` and the deploy.prototxt file ``` PlantClef_VGGmultipath_deploy  ``` are provided in the 'CNN' folder to run the code.





## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to ``` adeline87lee@gmail ``` or  ``` herve.goeau@cirad.  ```

## Lisense
BSD-3, see ``` LICENSE ``` file for details.
