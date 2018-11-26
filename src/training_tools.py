__author__ = "Adrián Pallas Fernández"
__email__ = "adrian.pallas@aimen.es"


from ConvLMD import cyplam_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import tables

def plot_boxes(labels,predicted,name=''):

    n = len(labels)

    lbls = []
    data = []

    for i in range(n):
        if len(labels[i])== 0:
            continue
        lbls.append(np.mean(labels[i]))
        data.append(predicted[i])

    plt.title('Test set prediction per label')
    plt.xlabel('true ' + name)
    plt.ylabel('predicted ' + name)
    plt.boxplot(data,showfliers=False)
    plt.xticks(range(1,len(lbls)+1),lbls)

    plt.savefig('./boxes_' + name + '.pdf')
    plt.close()

def plot_mean_std(labels,predicted,name=''):

    n = len(labels)

    lbls = []
    means = []
    stds = []

    for i in range(n):
        if len(labels[i])== 0:
            continue
        lbls.append(np.mean(labels[i]))
        means.append(np.mean(predicted[i]))
        stds.append(np.std(predicted[i]))

    lbl = np.array(lbls).ravel()
    means = np.array(means).ravel()
    stds = np.array(stds).ravel()

    plt.title('Test set prediction per label')
    plt.xlabel('true ' + name)
    plt.ylabel('predicted ' + name)
    #plt.errorbar([1,2,3,4],[1,2,3,4],yerr=[0.1,0.2,0.3,0.4],fmt='-o')


    plt.errorbar(lbl,means,yerr=stds,fmt='o')
    plt.plot(lbl,lbl,color='green')

    plt.savefig('./mean_std_' + name + '.pdf')
    plt.close()



def plot_errorbar(labels,predicted,name=''):

    data = []
    means = []
    stds = []

    for p in predicted:

        means.append(np.mean(p))
        stds.append(np.std(p))
        data.append(p)

    plt.title('Test set prediction per label')
    plt.xlabel('true ' + name)
    plt.ylabel('predicted ' + name)
    plt.errorbar(labels,means,yerr=stds,fmt='o')
    plt.plot(labels,labels,color='green')

    plt.savefig('error_'+name+'.pdf')
    plt.close()



def leave_one_out(file_train, id_name, x, y, batch_size=32):

    table = tables.open_file(file_train, mode='r').get_node('/df')

    ids = table.read(field=id_name)

    unique_ids = np.unique(ids)

    y_values = []
    y_predictions = []

    for id in unique_ids:

        model = cyplam_model(input_shape=(28, 28, 1), classes=1)

        model.compile(loss='mean_squared_error',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-5),
                      metrics=['mae','accuracy'])

        train_set_x = np.asarray(np.array([row[x] for row in table.where('(' + id_name + ' != ' + str(id) + ')')]),
                                 np.float32)[:,2:30,2:30] / 255
        train_set_y = np.array([row[y] for row in table.where('(' + id_name + ' != ' + str(id) + ')')])

        val_set_x = np.asarray(np.array([row[x] for row in table.where('(' + id_name + ' == ' + str(id) + ')')]),
                    np.float32)[:,2:30,2:30] / 255
        val_set_y = np.array([row[y] for row in table.where('(' + id_name + ' == ' + str(id) + ')')])

        train_datagen = ImageDataGenerator()

        val_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(
            train_set_x,
            train_set_y,
            batch_size=batch_size)

        val_generator = val_datagen.flow(
            val_set_x,
            val_set_y,
            batch_size=batch_size)

        save_callback = ModelCheckpoint(
            './' + y.__str__() + '.hdf5',
            monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        model.fit_generator(
            train_generator,
            callbacks=[save_callback],
            samples_per_epoch=len(train_set_y),
            epochs=20,
            validation_data=val_generator,
            nb_val_samples=len(val_set_y))

        model = load_model('./' + y.__str__() + '.hdf5')

        predictions = model.predict(val_set_x,32)

        y_values.append([val_set_y[0]])
        y_predictions.append(np.array(predictions).ravel())


    plot_errorbar(y_values,y_predictions,y)

    return


def training(file_train,file_val,id_name,x,ys,batch_size=32):


    table_train = tables.open_file(file_train, mode='r').get_node('/df')
    table_val = tables.open_file(file_val, mode='r').get_node('/df')

    train_set_x = np.asarray(np.array([row[x] for row in table_train]),np.float32)[:, 2:30, 2:30] / 255
    train_set_y = np.array([[row[y] for y in ys] for row in table_train])


    val_set_x = np.asarray(np.array([row[x] for row in table_val]),np.float32)[:, 2:30, 2:30] / 255
    val_set_y = np.array([[row[y] for y in ys] for row in table_val])

    ids_val = table_val.read(field=id_name)

    unique_ids_val = np.unique(ids_val)
    unique_ys_val = np.unique(val_set_y)

    model = cyplam_model(input_shape=(28, 28, 1), classes=len(ys))

    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-5),
                  metrics=['mae', 'accuracy'])

    train_datagen = ImageDataGenerator()

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        train_set_x,
        train_set_y,
        batch_size=batch_size)

    val_generator = val_datagen.flow(
        val_set_x,
        val_set_y,
        batch_size=batch_size)

    save_callback = ModelCheckpoint(
        './' + ys.__str__() + '.hdf5',
        monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


    model.fit_generator(
        train_generator,
        callbacks=[save_callback],
        samples_per_epoch=len(train_set_y),
        epochs=20,
        validation_data=val_generator,
        nb_val_samples=len(val_set_y))

    model = load_model('./' + ys.__str__() + '.hdf5')

    predictions = model.predict(val_set_x,32)


    for i in range(len(ys)):

        y = ys[i]

        pred_i = predictions[:,i]

        val_set_y_i = val_set_y[:,i]

        pred = np.array([pred_i[np.where(val_set_y_i==j)] for j in unique_ys_val])
        labels = np.array([val_set_y_i[np.where(val_set_y_i==j)] for j in unique_ys_val])

        plot_mean_std(labels,pred,y)
        plot_boxes(labels,pred,y)



def height():

    file_train = '/media/adrian/DATOS/cyplam_dataset/train_height.hdf5'

    y = 'height'

    leave_one_out(file_train, 'id_recording', 'image', y)




if __name__ == '__main__':

    pass
