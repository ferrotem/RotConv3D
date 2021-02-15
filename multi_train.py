#%%
import os
GPU = "0,1,2"
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
strategy = tf.distribute.experimental.CentralStorageStrategy()#MirroredStrategy()
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
#%%
# tf.config.threading.set_inter_op_parallelism_threads(20)
from network import Model
from data_loaderF import Data_Loader
from utils import *

#%%
# model_Rot3D  = Model().network
# %%
# tf.keras.utils.plot_model(model_Rot3D, 'model_Rot3D.png', show_shapes=True)
#%%

import os
tf.summary
log_dir="logs/"
os.makedirs(log_dir, exist_ok=True)
summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'model={}__'.format(cfg.CLASSIFIER) + datetime.datetime.now().strftime("%m-%d-%H-%M")+"/train/")
val_summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + 'model={}__'.format(cfg.CLASSIFIER) + datetime.datetime.now().strftime("%m-%d-%H-%M")+"/validation/")

checkpoint_dir = 'training_checkpoints_{}'.format(cfg.CLASSIFIER)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
print("checkpoint_prefix", checkpoint_prefix)
os.makedirs(checkpoint_prefix, exist_ok=True)


BATCH_SIZE = 2*strategy.num_replicas_in_sync

# loss function, optimizer
with strategy.scope():
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = False,reduction=tf.keras.losses.Reduction.NONE) #CategoricalCrossentropy
    def compute_loss(labels, predictions):
        per_example_loss =loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=BATCH_SIZE)

with strategy.scope():
    t_recall = tf.keras.metrics.Recall(name="train_recall")
    v_recall = tf.keras.metrics.Recall(name="val_recall")
    
    optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    model_Rot3D  = Model().network
    tf.keras.utils.plot_model(model_Rot3D, 'model_Rot3D.png', show_shapes=True)
    checkpoint = tf.train.Checkpoint(siamese_optimizer=optimizer,model = model_Rot3D)


#%%

def train_step(inputs):
    batch_X,  labels = inputs
    # batch_M_exp = tf.expand_dims(batch_M, -1)
    # batch_X = tf.concat([batch_X, batch_M_exp], axis=-1)
    with tf.GradientTape(persistent=False) as tape:
        output,  Z,Y,X, d_out  = model_Rot3D([batch_X], training=True)
        loss = compute_loss(labels, output)
        
        gradients = tape.gradient(loss, model_Rot3D.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model_Rot3D.trainable_weights))
        
        t_recall.update_state(labels, output)
        
        return output, loss, labels, Z,Y,X, d_out, batch_X

def val_step(inputs):
    batch_X,  labels = inputs
    # batch_M_exp = tf.expand_dims(batch_M, -1)
    # batch_X = tf.concat([batch_X, batch_M_exp], axis=-1)
    
    output,  Z,Y,X, d_out  = model_Rot3D([batch_X], training=False)
    loss = compute_loss(labels, output)
    v_recall.update_state(labels, output)
    return output, loss, labels


@tf.function()
def distributed_train_step(dataset_inputs):
    per_replica_output, per_replica_losses, per_replica_labels, Z,Y,X, d_out, batch_X = strategy.run(train_step, args=(dataset_inputs,))
    output = strategy.experimental_local_results(per_replica_output)
    labels = strategy.experimental_local_results(per_replica_labels)
    Z_save = strategy.experimental_local_results(Z)
    Y_save = strategy.experimental_local_results(Y)
    X_save = strategy.experimental_local_results(X)
    D_save = strategy.experimental_local_results(d_out)
    B_input = strategy.experimental_local_results(batch_X)
    loss =strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
    return output, labels,  loss, Z_save,Y_save,X_save, D_save, B_input

@tf.function()
def distributed_val_step(dataset_inputs):
    per_replica_output, per_replica_losses, per_replica_labels = strategy.run(val_step, args=(dataset_inputs,))
    output = strategy.experimental_local_results(per_replica_output)
    labels = strategy.experimental_local_results(per_replica_labels)
    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)
    return output, labels,  loss
#%%
ds = Data_Loader()
train_ds, val_ds  = ds.train_ds.batch(BATCH_SIZE), ds.val_ds.batch(BATCH_SIZE)
train_list, val_list = ds.train_list, ds.val_list
print(len(train_list), len(val_list))

train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
val_dist_dataset = strategy.experimental_distribute_dataset(val_ds.take(1000))

#%%
def fit(epochs):
    step = 0
    pbar_epoch = tqdm(total = epochs, desc = "epochs")

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        pbar_steps = tqdm(total = len(train_list), desc =" train_steps")
        
        for x in train_dist_dataset:
            output,labels,  train_loss, Z,Y,X, d_out, B_input = distributed_train_step(x)
            # print("length: ", len(output))
            output, labels = output[0].numpy(), labels[0].numpy()
            
            Z_save_np,Y_save_np,X_save_np, D_save_np = Z[0].numpy(),Y[0].numpy(),X[0].numpy(), d_out[0].numpy()
            # print("Z-save shape: ", Z_save_np.shape)

            if step%30000==0:
                top_k = len(np.where(labels[0]==1))
                top5_out = output[0].argsort()[-top_k:][::-1]
                top5_labels = labels[0].argsort()[-top_k:][::-1]
                count = 0
                for el in top5_out:
                    if el in top5_labels:
                        count+=1
                accuracy = (count/top_k)
                train_recall  =t_recall.result()# strategy.reduce(tf.distribute.ReduceOp.SUM, t_recall.result(),axis=None)
                print("top_k: ", top_k, top5_out,top5_labels )
                print("accuracy: ", accuracy)
                print('loss: ', train_loss.numpy() )
                print('recall: ', train_recall.numpy())
                with summary_writer.as_default():
                    tf.summary.experimental.set_step(step)
                    tf.summary.scalar('loss', np.sum(train_loss.numpy()), step=step)  
                    tf.summary.scalar('recall', train_recall, step=step)
                    tf.summary.scalar('accuracy_top_5', accuracy, step=step)
                

                    np.save(f"./results/output/Z_{step}.npy",Z_save_np[0])
                    np.save(f"./results/output/Y_{step}.npy",Y_save_np[0])
                    np.save(f"./results/output/X_{step}.npy",X_save_np[0])
                    np.save(f"./results/output/D_{step}.npy",D_save_np[0])
                    np.save(f"./results/output/input_{step}.npy",B_input[0][0].numpy())
                    np.save(f"./results/output/output_{step}.npy",output[0])
                    
                    
                    # tf.summary.histogram('output', output)
                
                # np_out = np.concatenate([np_out, output], axis=0)
                # # # print("np_out_shape:", np_out.shape)
                # Z_out = np.concatenate([Z_out, Z.numpy()], axis=0)
                # Y_out = np.concatenate([Y_out, Y.numpy()], axis=0)
                # X_out = np.concatenate([X_out, X.numpy()], axis=0)
                # label_out = np.concatenate([label_out, labels], axis=0)
                # dense_out = np.concatenate([dense_out, d_out.numpy()], axis=0)

                # # print("label_out:", label_out.shape)
                
            step+=1*BATCH_SIZE
            
            if step%12000==0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            
            pbar_steps.update(BATCH_SIZE)
        pbar_steps.close()
        val_accuracy = []
        val_recall = []
        pbar_val = tqdm(total = len(val_list), desc =" val steps")
        for x in val_dist_dataset:
            output,labels, train_loss = distributed_val_step(x)
            # print("we reached this point")
            output, labels = output[0].numpy(), labels[0].numpy()            
            top_k = len(np.where(labels[0]==1))
            top5_out = output[0].argsort()[-top_k:][::-1]
            top5_labels = labels[0].argsort()[-top_k:][::-1]
            count = 0
            for el in top5_out:
                if el in top5_labels:
                    count+=1
            accuracy = (count/top_k)
            val_accuracy.append(accuracy)
            recall  = v_recall.result()
            # train_recall  = strategy.reduce(tf.distribute.ReduceOp.SUM, v_recall.result(),axis=None)
            val_recall.append(recall.numpy())
            pbar_val.update(1)

        with val_summary_writer.as_default():
            tf.summary.scalar('recall', np.mean(val_recall), step=step)  
            tf.summary.scalar('accuracy_top_5', np.mean(val_accuracy), step=step)
        pbar_val.close()
        pbar_epoch.update(1)
        checkpoint.save(file_prefix = checkpoint_prefix)
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
# %%
