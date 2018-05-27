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
  e0: val_acc=0.9301999956 *! bsf=0.9301999956  Saved 
  e1: val_acc=0.9537999988 *! bsf=0.9537999988  Saved 
  e2: val_acc=0.9638000041 *! bsf=0.9638000041  Saved 
  e3: val_acc=0.9634000033    bsf=0.9638000041 
  e4: val_acc=0.9638999969 *! bsf=0.9638999969  Saved 
  e5: val_acc=0.9706999958 *! bsf=0.9706999958  Saved 
  e6: val_acc=0.9726999968 *! bsf=0.9726999968  Saved 
  e7: val_acc=0.9731000006 *! bsf=0.9731000006  Saved 
  e8: val_acc=0.9724000096    bsf=0.9731000006 
  e9: val_acc=0.9734000057 *! bsf=0.9734000057  Saved 
  e10: val_acc=0.9739000052 *! bsf=0.9739000052  Saved 
  e11: val_acc=0.9711999983    bsf=0.9739000052 
  e12: val_acc=0.9712000042    bsf=0.9739000052 
  e13: val_acc=0.9754000008 *! bsf=0.9754000008  Saved 
  e14: val_acc=0.9736000001    bsf=0.9754000008 
  e15: val_acc=0.9750000030    bsf=0.9754000008 
  e16: val_acc=0.9754000008    bsf=0.9754000008 
  ```

