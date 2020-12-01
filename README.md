# Face-Recognizer
Face recognition program built using Local Binary Patters Histograms. Included is a labeled training set of pictures of 3 different people. The program is built to separate these people from one another when presented with test images and identify who the person is in each picture.

You can choose to use either the python (.py) or the jupyter notebook (.ipynb) file, both are included in this repository

#IMPORTANT!!
Please make sure to change the filepaths("C:\\...) for cascade_path, path_train, and path_test to their corresponding locations in your computer once you download them or else the program will not work, the filepaths curently in the code are for my own computer. I have included comments where those changes are necessary and I have also included where to find those files.


Included in the Face Recognition folder:


1) haar-cascade files that are used to detect the eyes, nose, and frontal face of a person in photos

2) Train folder with 15 labeled images of each of three characters from my favorite show, The Office :)

3) Test folder containing 3 unlabeled image that the face_recognizer must classify into one of the three people based on the training images.

When you run this code, you will get an output window, which displays the predicted outputs for test images. You can press the Space or X button to keep looping. There are three different people in the test images. The output for the first person, Dwight, looks like the following:


