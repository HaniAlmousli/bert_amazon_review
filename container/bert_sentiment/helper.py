
import tensorflow as tf
from official.nlp import optimization  # to create AdamW optmizer

def get_optimizer(epochs=1,train_ds_size=1,init_lr=1e-3,optimizer_type='adamw'):
    
    steps_per_epoch = train_ds_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type=optimizer_type)
    return optimizer

def _loss_function(real, pred):
    loss_object    = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, pred)
    return tf.reduce_sum(loss_)    

def _get_metrics():
    
    train_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    valid_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    return train_acc,train_loss,valid_acc,valid_loss

def _reset_stats(lst_metrics):
    for m in lst_metrics:
        m.reset_states()
    
 
