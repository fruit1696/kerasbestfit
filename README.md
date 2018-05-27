**Introducing kerasbestfit**

This is a Python module that I wrote to make training and finding the best Keras model easier. The module uses Keras's EarlyStopping and Checkpoint callbacks so that you just need to call one function and it finds and saves the model that has the best metrics.

The advantage to using this module is that you can just let it run and you'll find the best model in your folder where you can later use it for predictions. It can also stop after a certain duration so that you can fit the training session into your provider time limit. There is even a snifftest parameter where you can skip training an iteration that has very poor performance, thus saving training time.

Why did I create a new module instead of using Sklearn, T-Pot or something else to find the best fit? Because it made me dig into Keras to understand it better. Simple as that. The result is a pretty nifty function.

**Installation**

pip install kerasbestfit

**Usage**
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
                                 epochs=10,
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
                                 
**Output**   
  e0: val_acc=0.9293000042 *! bsf=0.9293000042  Saved 
  e1: val_acc=0.9520000011 *! bsf=0.9520000011  Saved 
  e2: val_acc=0.9629000008 *! bsf=0.9629000008  Saved 
  e3: val_acc=0.9635999948 *! bsf=0.9635999948  Saved 
  e4: val_acc=0.9630000025    bsf=0.9635999948 
  e5: val_acc=0.9680999964 *! bsf=0.9680999964  Saved 
  e6: val_acc=0.9713000029 *! bsf=0.9713000029  Saved 
  e7: val_acc=0.9719000012 *! bsf=0.9719000012  Saved 
  e8: val_acc=0.9738000005 *! bsf=0.9738000005  Saved 
  e9: val_acc=0.9747000039 *! bsf=0.9747000039  Saved 

