# gazetracker

Mobile gaze tracker implementations as part of a final year project. Built using Keras.

Based on work by [GazeCapture](http://gazecapture.csail.mit.edu). We use the dataset collected by Krafka et al to experiment with gaze trackers on mobile devices. We obtained validation errors of 3.230cm on our final model.

The models that we experimented with included the original ITracker, an Improved [modification](https://github.com/hugochan/Eye-Tracker), a model using [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) and a novel method attempting to use landmarks as an auxiliary network (MobileIFTracker). Using these methods, we obtained the following results on the validation set:

| Model                 | Loss / cm     |
| ----------------------|---------------|
| ITracker              | 3.638         |
| Improved              | 3.614         |
| SE-ResNet-ITracker    | 3.546         |
| SE-Mobile-ITracker    | 3.230         |
| MobileIFTracker       | 5.577         |
| SE-MobileIFTracker    | 5.230         |

Note that these results do not reflect accurately the implementations discussed in the [original paper](http://gazecapture.csail.mit.edu) and the [modification](https://github.com/hugochan/Eye-Tracker) suggested since we did not fine-tune the models due to lack of computation resources. Instead, we did find that SE networks do have interesting promise in improving the performance of the eye tracker.

## Dataset

The original dataset can be downloaded from [here](http://gazecapture.csail.mit.edu/download.php). The link appears to go down every now and then.

After obtaining the raw dataset, you have to pre-process the dataset to obtain crops of the face and eyes in order to run the training of the models. To do so, you can run the `prepareDataset.py` file from the source code [provided](https://github.com/CSAILVision/GazeCapture/tree/master/pytorch) by GazeCapture. This will produce the `metadata.mat` file that is used with `ITrackerData` module to load the images in each batch. 

To use the MobileIFTracker architectures, you will need the annotated landmarks `landmarks_metadata.mat`. The landmarks were annotated using [FAN](https://github.com/1adrianb/face-alignment) and [ELG](https://github.com/swook/GazeML). The whole process took 5 days and the pre-annotated landmarks `landmarks_metadata.mat` can be obtained by requesting from <jw.ziggee@gmail.com>. The file with the annotations was too big to be uploaded to Github.

Edit the `utils/ITrackerData.py` file to point to the necessary files and folder

To use the phones-only dataset, also request for `phone_metadata.mat` from <jw.ziggee@gmail.com>. Or modify the prepareDataset.py from ITrackerData to exclude files that have orientation `2`.

## Installing dependencies

Install the required dependencies using:

```shell
pip install -r requirements.txt
```

## Training

You can execute training by running the following command:

```shell
python train.py --model <MODEL_TO_USE> --epochs <EPOCHS> [OPTIONS]

Options:
--model     Model to use (baseline, improved, seresnet, semobile, mobileift, semobileift)
--epochs    Number of epochs to train (default: 1)
--aug       Augmentations to use: none (default), brightness, erasing
--weights   Path to weights to be loaded to start training from (optional)
```

The model weights will then be saved to the `models/output` folder.

## Testing

You can test your model using:

```shell
python test.py --model <MODEL_TO_USE> --weights <PATH_TO_WEIGHTS>

Options:
--model     Model to use (baseline, improved, seresnet, semobile, mobileift, semobileift)
--weights   Path to weights to be loaded to evaluate model at
```

## Pre-trained Models

Pre-trained models are not available because of re-structuring of code for submission. Reported results should however be similar if trained.

## Credits

This repository uses many implementations from open-source libraries such as the [Keras implementation of SENet](https://github.com/yoheikikuta/senet-keras), [random erasing](https://github.com/yu4u/cutout-random-erasing) and of course from the GazeCapture [source code](https://github.com/CSAILVision/GazeCapture).
