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

3. Users can train the model from scratch by running the ``` mainClef.py ``` which includes ``` temp_createStruct5.py ``` and ``` attn_7_1_ex.py ```.

4. Users can test the trained model prepared in the 'model' folder.



Note that users are expected to modify the corresponding files to correct path to work properly. Enjoy!

## Feedback
Suggestions and opinions of this work (both positive and negative) are greatly welcome. Please contact the authors by sending email to ``` adeline87lee@gmail ``` or  ``` herve.goeau@cirad.  ```

## Lisense
BSD-3, see ``` LICENSE ``` file for details.
