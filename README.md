# ExtremelyDarkImages_DeepLearning
Group30_DeepLearning

We decided to use the Sony dataset from 'Learning to See in the Dark'. This is one of the original datasets used by the paper. The dataset consists of 'short exposure' images (classified into training, testing and validation) and 'long exposure' images which can be used as the 'ground truth' of the short exposure ones. The original code contained little information on how to properly load the images and was missing the references to the coding system (where the first digit of the name indicates if the image is training(0), testing (1) or validation(2)). Note that we used the 0.1s exposure images, and not the 0.04s ones. Here is the code we adapted wrote to extract the images properly:

```
train_files = glob.glob('C:/path_to_folder/Sony/short/0*_00_0.1s.ARW') #training files
train_files += glob.glob('C:/path_to_folder/Sony/short/2*_00_0.1s.ARW') #validation files

gt_files = []
for x in train_files:
    gt_files += glob.glob('C:/path_to_folder/Sony/long/*'+x[-17:-12]+'*.ARW') #GT files for training/validation

test_files = glob.glob('C:/path_to_folder/Sony/short/1*_00_0.1s.ARW') #test files
for x in test_files:
    gt_files +=glob.glob('C:/path_to_folder/Sony/long/*'+x[-17:-12]+'*.ARW') #GT files for test
```

Note that for performing a 'dry run' the lists must be size accordingly. This is done through the variables 'dry_run_trainpictures' and 'dry_run_testpictures'