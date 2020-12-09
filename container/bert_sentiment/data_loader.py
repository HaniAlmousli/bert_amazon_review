
import numpy as np
import tensorflow as tf
import tensorflow_text as text

def get_tr_va_te(data_path, batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    seed = 42

    raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
        data_path+'/training',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

    
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.keras.preprocessing.text_dataset_from_directory(
        data_path+'/training',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #**** To be implemeted
    # test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    #     data_path+'/testing',
    #     batch_size=batch_size)

    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return (train_ds,val_ds,[])