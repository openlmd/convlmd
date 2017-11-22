

import csv
import numpy as np
import tf_cnn_linear2
import tensorflow as tf
import argparse
import os

DATASET_FOLDER = os.path.join(os.getcwd(),'DATASET')

def data_setup(frames,labels):

    train_idx = labels>0
    val_idx = labels<0

    train_frames = frames[train_idx]
    val_frames = frames[val_idx]
    train_id = np.where(train_idx)[0]

    train_labels = labels[train_idx]
    val_labels = (labels[val_idx]*-1)
    val_id = np.where(val_idx)[0]

    train_id = np.repeat(train_id,train_frames.shape[1])
    train_labels = np.repeat(train_labels,train_frames.shape[1])

    val_id = np.repeat(val_id,val_frames.shape[1])
    val_labels = np.repeat(val_labels,val_frames.shape[1])

    train_frames = train_frames.reshape(train_frames.shape[0]*train_frames.shape[1],train_frames.shape[2])

    idx = range(train_frames.shape[0])

    np.random.shuffle(idx)

    train_frames = train_frames[idx]
    train_labels = train_labels[idx]
    train_id = train_id[idx]

    val_frames = val_frames.reshape(val_frames.shape[0]*val_frames.shape[1],val_frames.shape[2])
    selected_targets = np.unique(train_labels)

    targets_mask = np.zeros(len(train_labels))

    masks=np.array([train_labels==v for v in selected_targets])

    min_values = np.min(np.array([np.sum(m) for m in masks]))

    for i in range(len(masks)):

        idx = np.where(masks[i]==True)[0]
        np.random.shuffle(idx)
        targets_mask[idx[:min_values]]=1

    train_labels = train_labels[targets_mask!=0]
    train_frames = train_frames[targets_mask!=0]
    train_id = train_id[targets_mask!=0]

    return (train_frames,train_labels,train_id),(val_frames,val_labels,val_id)



def load_npy_data(dataset_folder,n_test):
    

    csv_file = open(str(dataset_folder)+os.sep+'cords.csv', 'r')

    reader = csv.reader(csv_file,delimiter=' ',)

    cords = np.array([[col for col in row] for row in reader])

    frames = [(np.load(dataset_folder+os.sep+dir+'.npy'))[2000:-2000] for dir in cords[:,0]]

    power = np.asarray(cords[:,1],np.float64)
    speed = np.asarray(cords[:,2],np.float64)

    min_frames = np.min([len(frame) for frame in frames])

    frames = np.array([frame[(len(frame)-min_frames)//2:(len(frame)+min_frames)//2] for frame in frames])[:,:,2:30,2:30]

    labels = np.array([float(power[i]*100+speed[i]) for i in range(len(power))])

    frames = np.asarray(frames.reshape(frames.shape[0],-1,784),np.float32)/1024.

    idx = range(len(labels))

    np.random.shuffle(idx)

    idx=idx[:n_test]

    labels[idx]=-labels[idx]


    return data_setup(frames,labels)


def main():

    TRAINING_ITERS = 150000
    BATCH_SIZE = 128
    n_input=28*28
    n_classes = 2
    DISPLAY_STEP = 100
    dropout = 0.75
    seed = 13

    tf.set_random_seed(seed)
    np.random.seed(seed)
    
    (features, targets, ids) ,(val_features ,val_targets, val_ids) = load_npy_data(DATASET_FOLDER,10)

    targets = np.transpose(np.vstack((targets//100,targets%100)))
    val_targets = np.transpose(np.vstack((val_targets//100,val_targets%100)))


    X_train = features
    y_train = targets

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32)

    pred = tf_cnn_linear2.model(x, y, keep_prob)
    accuracy = tf_cnn_linear2.evaluation(pred, y)
    paccuracy = tf_cnn_linear2.power_evaluation(pred, y)
    saccuracy = tf_cnn_linear2.speed_evaluation(pred, y)
    cost, optimizer = tf_cnn_linear2.training(pred, y)


    with tf.Session() as sess:

        print('session initiated')

        sess.run(tf.global_variables_initializer())

        step = 1

        saver = tf.train.Saver()

        while step * BATCH_SIZE < TRAINING_ITERS:

            idx = np.random.randint(X_train.shape[0],size=BATCH_SIZE)

            batch_x = X_train[idx]
            batch_y = y_train[idx]

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

            if step % DISPLAY_STEP == 0:

                loss, acc = sess.run([cost,accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})


                print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " +
                        "{:.6f}".format(loss) + ", Training Accuracy= " +
                        "{:.5f}".format(acc))

            step += 1
        print("Optimization Finished!")



        saver.save(sess, "model.ckpt")

        predicted2 = None

        val_accuracy=0

        for i in range(len(val_targets)//256):

            value,acc = sess.run([pred,accuracy], feed_dict={x: val_features[i*256:(i+1)*256],
                                                             y: val_targets[i*256:(i+1)*256],
                                                             keep_prob: 1.})
            val_accuracy+=acc

            if predicted2 is None:
                predicted2 = value
            else:
                predicted2 = np.vstack((predicted2,value))

        val_accuracy/=(len(val_targets)//256)
        print('VAL ACCURACY',val_accuracy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to de image")
    args = parser.parse_args()

    DATASET_FOLDER = args.path
    DATASET_FOLDER = os.path.join(os.getcwd(),'DATASET')

    main()