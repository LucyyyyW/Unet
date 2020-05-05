# Unet
The LGE MRI used in the study were collected from 45 patients, of which 9 were randomly selected for testing. To augment the training data, I registered the training images to other image spaces using a set of artificially generated rigid, affine and deformable transformations, resulting in 5405 2D slices.
I used Dice coefficient as metrics for evaluation of segmentation accuracy. The Dice of LV blood pool, Dice of Myocardium and Dice of RV blood pool on test data have reached 0.90,0.81 and 0.83 respectively.
