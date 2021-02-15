#%%
import os
GPU = "3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU

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

# tf.config.threading.set_inter_op_parallelism_threads(20)
from network_test import Model
from data_loader import Data_Loader
from utils import *


#%%

import os
tf.summary
log_dir="logs/"
os.makedirs(log_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'model={}__'.format( cfg.CLASSIFIER) + datetime.datetime.now().strftime("%m-%d-%H-%M")+"/train/")
val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'model={}__'.format(cfg.CLASSIFIER) + datetime.datetime.now().strftime("%m-%d-%H-%M")+"/validation/")


# loss function, optimizer
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = False) #CategoricalCrossentropy
recall_object = tf.keras.metrics.Recall()
optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = 'training_checkpoints_{}'.format(cfg.CLASSIFIER)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print("checkpoint_prefix", checkpoint_prefix)
os.makedirs(checkpoint_prefix, exist_ok=True)
latest = tf.train.latest_checkpoint(checkpoint_dir)
print("latest: ", latest)
model_Rot3D  = Model().network
checkpoint = tf.train.Checkpoint(model = model_Rot3D)
checkpoint.restore(checkpoint_dir + '/ckpt-8')

# model_Rot3D.load_weights(latest)


#%%

def unbalance_softsign_loss(labels, logits):
    gamma = 1.25 *labels - 0.25 
    res = 1 - tf.math.log1p( gamma*logits/(1+ tf.math.abs(logits)) )

    return res 

@tf.function()
def train_step(batch_clip,batch_M,  labels, training = True):
    batch_M_exp = tf.expand_dims(batch_M, -1)
    batch_clip = tf.concat([batch_clip, batch_M_exp], axis=-1)
    with tf.GradientTape(persistent=False) as tape:

        # Calling our siamese model
        output,  Z,Y,X, d_out  = model_Rot3D([batch_clip], training=training)
        # output = tf.squeeze(output)
        # tf.print("ouput: ", tf.math.reduce_sum(output))
        # tf.print("labels: ", labels.shape)
        # tf.print("outputs: ", output.shape)
        # tf.print("label sum", tf.math.reduce_sum(labels))
        # BinaryCrossentropy loss from logits 
        loss = loss_object(labels, output)
        #print("we reached this point")
        # tf.print("loss: ", loss)
        gradients = tape.gradient(loss, model_Rot3D.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model_Rot3D.trainable_weights))
        recall_object.update_state(labels, output)
        recall  = recall_object.result()
        return output, loss, recall, Z,Y,X,d_out,  batch_clip


#%%
ds = Data_Loader()
train_ds, val_ds  = ds.train_ds, ds.val_ds
train_list, val_list = ds.train_list, ds.val_list

#%%
def fit(epochs):
    step = 0
    pbar_epoch = tqdm(total = epochs, desc = "epochs")
    np_out = np.zeros((1,80))
    Z_out = np.zeros((6,54,54,32))
    Y_out = np.zeros((6,54,54,32))
    X_out = np.zeros((6,54,54,32))
    dense_out = np.zeros((1,2048))
    label_out = np.zeros((1,80))
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        pbar_steps = tqdm(total = len(train_list), desc =" train_steps")
        
        for _, (batch_X, batch_M, batch_Y) in train_ds.batch(1).enumerate():
            output, train_loss, train_recall, Z,Y,X ,d_out,batch_clip = train_step(batch_X, batch_M, batch_Y, True)
            # print("we reached this point")
            output, labels = output.numpy(), batch_Y.numpy()
            # labels = batch_Y.numpy()
            # print("output shape: ", output.shape)
            # top_5 = output[0].argsort()[-5:][::-1]
            # count = 0
            # label_set = (labels[0] == 1).nonzero()#np.where(labels.cpu().numpy() == 1).
            # # print("labelset:", label_set)
            # # print("top5 set:", top_5)
            # for i in range(len(label_set)):
            #     if label_set[i] in top_5:
            #         count+=1
            # accuracy = (count/5)
            np.save(f"./results/output/input_{step}.npy", batch_clip.numpy())
            np.save(f"./results/output/Z_{step}.npy",Z.numpy())
            np.save(f"./results/output/Y_{step}.npy",Y.numpy())
            np.save(f"./results/output/X_{step}.npy",X.numpy())
            np.save(f"./results/output/D_out_{step}.npy",d_out.numpy())
            if step>3:
                break


            if step%1000==0:
                top_k = len(np.where(labels[0]==1))
                top5_out = output[0].argsort()[-top_k:][::-1]
                top5_labels = labels[0].argsort()[-top_k:][::-1]
                count = 0
                for el in top5_out:
                    if el in top5_labels:
                        count+=1
                accuracy = (count/top_k)
                print("top_k: ", top_k, top5_out,top5_labels )
                print("accuracy: ", accuracy)
                print('loss: ', train_loss.numpy() )
                print('recall: ', train_recall.numpy())
                with summary_writer.as_default():
                    tf.summary.experimental.set_step(step)
                    tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=step)  
                    tf.summary.scalar('recall', train_recall, step=step)
                    tf.summary.scalar('accuracy_top_5', accuracy, step=step)
                    # tf.summary.histogram('output', output)
                
                # np_out = np.concatenate([np_out, output], axis=0)
                # # # print("np_out_shape:", np_out.shape)
                # Z_out = np.concatenate([Z_out, Z.numpy()], axis=0)
                # Y_out = np.concatenate([Y_out, Y.numpy()], axis=0)
                # X_out = np.concatenate([X_out, X.numpy()], axis=0)
                # label_out = np.concatenate([label_out, labels], axis=0)
                # dense_out = np.concatenate([dense_out, d_out.numpy()], axis=0)

                # print("label_out:", label_out.shape)
                
            step+=1
            pbar_steps.update(1)
        pbar_steps.close()
        val_accuracy = []
        val_recall = []
        pbar_val = tqdm(total = len(val_list), desc =" val steps")
        for _, (batch_X, batch_M, batch_Y) in val_ds.batch(1).enumerate():
            output, train_loss, train_recall, Z,Y,X,d_out,batch_clip  = train_step(batch_X, batch_M, batch_Y, False)
            # print("we reached this point")
            output, labels = output.numpy(), batch_Y.numpy()
            # print("output shape: ", output.shape)
            # top_5 = output[0].argsort()[-5:][::-1]
            # count = 0
            # label_set = (labels[0] == 1).nonzero()#np.where(labels.cpu().numpy() == 1).
            # # print("labelset:", label_set)
            # # print("top5 set:", top_5)
            # for i in range(len(label_set)):
            #     if label_set[i] in top_5:
            #         count+=1
            # accuracy = (count/5)
            
            top_k = len(np.where(labels[0]==1))
            top5_out = output[0].argsort()[-top_k:][::-1]
            top5_labels = labels[0].argsort()[-top_k:][::-1]
            count = 0
            for el in top5_out:
                if el in top5_labels:
                    count+=1
            accuracy = (count/top_k)
            val_accuracy.append(accuracy)
            val_recall.append(train_recall.numpy())
            pbar_val.update(1)

        with val_summary_writer.as_default():
            tf.summary.scalar('recall', np.mean(val_recall), step=step)  
            tf.summary.scalar('accuracy_top_5', np.mean(val_accuracy), step=step)
        pbar_val.close()
        pbar_epoch.update(1)
        # if epoch%1==0:
        #     np.save(f"./results/output/np_out_{epoch}.npy",np_out)
        #     np.save(f"./results/output/Z_{epoch}.npy",Z_out)
        #     np.save(f"./results/output/Y_{epoch}.npy",Y_out)
        #     np.save(f"./results/output/X_{epoch}.npy",X_out)
        #     np.save(f"./results/output/labels_{epoch}.npy",label_out)
        #     np.save(f"./results/output/dense_{epoch}.npy",dense_out)
#%%
from tqdm import tqdm
fit(100)
