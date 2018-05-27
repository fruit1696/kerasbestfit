**Introducing kerasbestfit**

This is a Python module that I wrote to make training and finding the best Keras model easier. The module uses Keras's EarlyStopping and Checkpoint callbacks so that you just need to call one function and it finds and saves the model that has the best metrics.

The advantage to using this module is that you can just let it run and you'll find the best model in your folder where you can later use it for predictions. It can also stop after a certain duration so that you can fit the training session into your provider time limit. There is even a snifftest parameter where you can skip training an iteration that has very poor performance, thus saving training time.

Why did I create a new module instead of using Sklearn, T-Pot or something else to find the best fit? Because it made me dig into Keras to understand it better. Simple as that. The result is a pretty nifty function.

**Installation**

pip install kerasbestfit

**Usage**
```python

# train with 20 epochs with patience of 3, save the model structure and weights, and show the best so far (bsf)

def log_msg(msg=''):
    print(msg)
    return
    
results, log = kbf.find_best_fit(model=model,
                                 metric='val_acc',
                                 xtrain=train_images,
                                 ytrain=train_labels,
                                 xval=test_images,
                                 yval=test_labels,
                                 validation_split=0,
                                 batch_size=500,
                                 epochs=20,
                                 patience=3,
                                 snifftest_max_epoch=0,
                                 snifftest_metric_val=0,
                                 show_progress=True,
                                 format_metric_val='{:1.10f}',
                                 save_best=True,
                                 save_path='',
                                 best_metric_val_so_far=0,
                                 logmsg_callback=log_msg,
                                 finish_by=0)
```                                 
                                 
**Output**   
```text
# here we can see that it aborted after 12 epochs since it went 3 epochs without a better result.
# best result is 0.9745000005

  e0: val_acc=0.9284999937 *! bsf=0.9284999937  Saved 
  e1: val_acc=0.9526999980 *! bsf=0.9526999980  Saved 
  e2: val_acc=0.9618999988 *! bsf=0.9618999988  Saved 
  e3: val_acc=0.9645999998 *! bsf=0.9645999998  Saved 
  e4: val_acc=0.9637999952    bsf=0.9645999998 
  e5: val_acc=0.9687000036 *! bsf=0.9687000036  Saved 
  e6: val_acc=0.9700999945 *! bsf=0.9700999945  Saved 
  e7: val_acc=0.9718000025 *! bsf=0.9718000025  Saved 
  e8: val_acc=0.9741999984 *! bsf=0.9741999984  Saved 
  e9: val_acc=0.9745000005 *! bsf=0.9745000005  Saved 
  e10: val_acc=0.9736999959    bsf=0.9745000005 
  e11: val_acc=0.9718000025    bsf=0.9745000005 
  e12: val_acc=0.9694999963    bsf=0.9745000005 
  ```

