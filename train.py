#%%
import tensorflow as tf 
import config as cfg
from matplotlib import pyplot as plt

import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from network import Model


#%%
model_Rot3D  = Model().network
# %%
tf.keras.utils.plot_model(model_Rot3D, 'model_Rot3D.png', show_shapes=True)
#%%

import os
# tf.summary
log_dir="tensorboard_logs/"
os.makedirs(log_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'model={}__'.format( cfg.CLASSIFIER) + datetime.datetime.now().strftime("%m-%d-%H-%M")+"/train/")
val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'model={}__'.format(cfg.CLASSIFIER) + datetime.datetime.now().strftime("%m-%d-%H-%M")+"/validation/")


# loss function, optimizer
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits = False) #BinaryCrossentropy
recall_object = tf.keras.metrics.Recall()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = 'training_checkpoints_{}'.format(cfg.CLASSIFIER)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print("checkpoint_prefix", checkpoint_prefix)
os.makedirs(checkpoint_prefix, exist_ok=True)

#%%
@tf.function()
def train_step(batch_clip, labels, training = True):
    with tf.GradientTape(persistent=False) as tape:

        # Calling our siamese model
        output  = model_Rot3D([batch_clip], training=training)
        # output = tf.squeeze(output)
        # tf.print("ouput: ", tf.math.reduce_sum(output))
        # tf.print("labels: ", labels.shape)
        # tf.print("outputs: ", output.shape)
        # tf.print("label sum", tf.math.reduce_sum(labels))
        # BinaryCrossentropy loss from logits 
        loss = loss_object(labels, output)
        #print("we reached this point")
        tf.print("loss: ", loss)
        gradients = tape.gradient(loss, model_Rot3D.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model_Rot3D.trainable_weights))
        recall_object.update_state(labels, output)
        recall  = recall_object.result()
        return output, loss, recall

#%%


#%%
from data_loader import Data_Loader
from utils import *
# annotation = file_reader(cfg.ANNOTATION_PATH)
# data_ratio = 0.7
# total_list = np.arange(len(annotation))
# np.random.shuffle(total_list)
# divider =round(len(annotation)*data_ratio)
# train_list, val_list = total_list[:divider], total_list[divider:]

# dataset = Dataset(annotation)


# def input_generator(id_list):
#     for idx in range(len(id_list)):
#         yield id_list[idx]

# train_ds = tf.data.Dataset.from_generator(input_generator , args= [train_list], output_types= (tf.int32))
# val_ds = tf.data.Dataset.from_generator(input_generator ,args=[val_list], output_types= (tf.int32))

ds = Data_Loader()
train_ds, val_ds  = ds.train_ds, ds.val_ds
train_list, val_list = ds.train_list, ds.val_ds
# def read_transform(idx):
#     [frame_list, label] = tf.py_function(dataset._single_input_generator, [idx], [tf.float32, tf.int32])
#     return frame_list, label


# train_ds =train_ds.map(read_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
# autotune = tf.data.experimental.AUTOTUNE
# train_ds = train_ds.prefetch(autotune)
# val_ds = val_ds.map(read_transform,num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
# val_ds = val_ds.prefetch(autotune)
#%%
# for [f, l] in train_ds.take(2):
#     print(f.shape)
#     print(l)
#     plt.imshow(f[0].numpy())
#     break
#%%
def fit(epochs):
    step = 0
    pbar_epoch = tqdm(total = epochs, desc = "epochs")
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        pbar_steps = tqdm(total = len(train_list), desc =" train_steps")

        for _, (batch_X, batch_Y) in train_ds.batch(1).enumerate():
            output, train_loss, train_recall = train_step(batch_X, batch_Y, True)
            # print("we reached this point")
            output, labels = output.numpy(), batch_Y.numpy()
            # print("output shape: ", output.shape)
            top_5 = output[0].argsort()[-5:][::-1]
            count = 0
            label_set = (labels[0] == 1).nonzero()#np.where(labels.cpu().numpy() == 1).
            # print("labelset:", label_set)
            # print("top5 set:", top_5)
            for i in range(len(label_set)):
                if label_set[i] in top_5:
                    count+=1
            accuracy = (count/5)


            if step%30==0:
                print("accuracy: ", accuracy)
                print('loss: ', train_loss.numpy() )
                print('recall: ', train_recall.numpy())
                with summary_writer.as_default():
                    tf.summary.experimental.set_step(step)
                    tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=step)  
                    tf.summary.scalar('recall', train_recall, step=step)
                    tf.summary.scalar('accuracy_top_5', accuracy, step=step)
                    tf.summary.histogram('output', output)
                
            step+=1
            pbar_steps.update(1)
        pbar_steps.close()
        val_accuracy = []
        val_recall = []
        pbar_val = tqdm(total = len(val_list), desc =" val steps")
        for _, (batch_X, batch_Y) in val_ds.batch(1).enumerate():
            output, train_loss, train_recall = train_step(batch_X, batch_Y, False)
            # print("we reached this point")
            output, labels = output.numpy(), batch_Y.numpy()
            # print("output shape: ", output.shape)
            top_5 = output[0].argsort()[-5:][::-1]
            count = 0
            label_set = (labels[0] == 1).nonzero()#np.where(labels.cpu().numpy() == 1).
            # print("labelset:", label_set)
            # print("top5 set:", top_5)
            for i in range(len(label_set)):
                if label_set[i] in top_5:
                    count+=1
            accuracy = (count/5)
            val_accuracy.append(accuracy)
            val_recall.append(train_recall.numpy())
            pbar_val.update(1)

        with val_summary_writer.as_default():
            tf.summary.scalar('recall', np.mean(val_recall), step=step)  
            tf.summary.scalar('accuracy_top_5', np.mean(val_accuracy), step=step)
        pbar_val.close()
        pbar_epoch.update(1)
#%%
from tqdm import tqdm
fit(100)
# %%
