
## The data

As you can see, all data can be found in the data folder. For your purposes, the data has already been split in training data and test data. They are respectively in the train folder and test folder. In those folders, you can find four folders which represent the mugs you'll need to classify. There are four kind of mugs: the white mug, the black mug (the ML6 mug), the blue mug and the transparent mug (the glass). The white mug is class 0, the black mug class 1, the blue mug class 2 and the transparent mug class 3. These class numbers are necessary to create a correct classifier. If you want, you can inspect the data, however, the code to load the data of the images into numpy arrays is already written for you.

## The model

In the trainer folder, you will be able to see several python files. The data.py, task.py and final_task.py files are already coded for you. The only file that needs additional code is the model.py file. The comments in this file will indicate which code has to be written.

To test how your model is doing you can execute the following command (you will need to [install](https://cloud.google.com/sdk/docs/) the gcloud command):

```
gcloud ml-engine local train --module-name trainer.task --package-path trainer/ -- --eval-steps 5
```

The command above will perform 5 evaluation steps during the training. If you want to change this, you only have to change the 5 at the end of the command to the number of evaluation steps you like. The batch size and the number of training steps should be defined in the model.py file.


![Data overview](data.png =1x)

The command above uses the task.py file. As you can see in the figure above, this file only uses the mug images in the training folder of this repository and uses the test folder to evaluate the model. This is excellent to test how the model performs but to obtain a better evaluation one can also train upon all available data which should increase the performance on the dataset you will be evaluated. After you finished coding up model.py, you can read on and you'll notice how to train your model on the full dataset.

## Deploying the model

Once you've got the code working you will need to deploy the model to Google Cloud to turn it into an API that can receive new images of mugs and returns its prediction for this mug. The code for this is already written in the final_task.py file. To deploy the model you've just written, you only have to run a few commands in your command line.

To export your trained model and to train your model on the training folder and the test folder you have to execute the following command (only do this once you've completed coding the model.py file):

```
gcloud ml-engine local train --module-name trainer.final_task --package-path trainer/
```

Once you've executed this command, you will notice that the output folder was created in the root directory of this repository. This folder contains your saved model that you'll be able to deploy on Google Cloud ML-engine.

To be able to deploy the model on a Google Cloud ML-engine you will need to create a [Google Cloud account](https://cloud.google.com/). You will need a credit card for this, but you'll get free credit from Google to run your ML-engine instance.

Once you've created your Google Cloud account, you'll need to deploy your model on a project you've created. You can follow a [Google guideline](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction) for this.

## Checking your deployed model

You can check if your deployed model works the way it should by executing the following commands:

```
MODEL_NAME=<your_model_name>
VERSION=<your_version_of_the_model>
gcloud ml-engine predict --model $MODEL_NAME --version $VERSION --json-instances check_deployed_model/test.json
```

Check if you are able to get a prediction out of the gcloud command. The output of the command should look something like this.

```
CLASSES  PROBABILITIES
1        [2.0589146706995187e-12, 1.0, 1.7370329621294728e-13, 1.2870057122347237e-32]
```

The values you use for the $MODEL_NAME variable and the $VERSION variable can be found in your project on the Google Cloud web interface. 

