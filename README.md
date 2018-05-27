**kerasbestfit**

This is a Python module that I wrote to make training and finding the best Keras model easier. The module uses Keras's EarlyStopping and Checkpoint callbacks so that you just need to call one function and it finds and saves the model that has the best metrics.

The advantage to using this module is that you can just let it run and you'll find the best model in your folder where you can later use it for predictions. It can also stop after a certain duration so that you can fit the training session into your provider time limit. There is even a snifftest parameter where you can skip training an iteration that has very poor performance, thus saving training time.

Why did I create a new module instead of using Sklearn, T-Pot or something else to find the best fit? Because it made me dig into Keras to understand it better. Simple as that. The result is a pretty nifty function.

**Installation**

pip install kerasbestfit
