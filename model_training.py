import numpy as np
import tensorflow as tf
import random as rn

###for reproducibility of results
np.random.seed(99)
rn.seed(111)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(123)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf) #

K.set_session(sess)

from keras.applications.vgg16 import VGG16
from keras import applications
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout, Concatenate, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.optimizers import Adam, SGD, RMSprop

from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.model_selection import KFold, cross_validate, GridSearchCV, train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics.cluster import entropy
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.utils import class_weight
from sklearn.utils import parallel_backend

import matplotlib.pyplot as plt
import os, sys, math, argparse, json, time, datetime
import nibabel as nib
from collections import Counter
import cv2
import pandas as pd

import classification_models
###################################################################################################################

FLAGS=None

def vol2img_indices(vol_indices, Nslices):
    img_indices=[]
    for i in range(len(vol_indices)):
        for j in range(Nslices):
            img_indices.append((vol_indices[i] * Nslices )+ j)

    img_indices=np.array(img_indices)
    return img_indices


def single2three_channel(data):
    """ converts single channel data in to three channel data.

    # Arguments:
        data: nd float array, single channel input array.

    # Returns
        data: nd float array, three channel output array.
    """
    data=np.concatenate([data,data,data],axis=-1)
    return data

def plot_histogram(array_data, save_path, Nslices, dataset_name):
    """plots the histogram of the array_data and saves the plot at the save_path

    # Arguments
        array_data: numpy array
        save_path: location to save the histogram plot
        Nslices: number of chosen MRI slices
        dataset_name: name of the dataset to use as a save name for the histogram plot.

    """
    b=array_data.ravel()#convert nd array to 1d
    c=Counter(b)
    plt.bar(c.keys(), c.values())
    for a,b in zip(c.keys(), c.values()):
         plt.text(a,b,str(b))

    plt.savefig(os.path.join(save_path, dataset_name+str(Nslices)+'_hist.png'))
    plt.show()



def choose_slice_indices(nifti_data, Naxis, Nslices):
    """ chooses Nslices from each 3D MRI volume that maximize the image entropy value

    # Arguments
        nifti_data: 4D numpy array data.
        Nslices: integer. Number of slices to choose from each volume.
        Naxis: integer. slicing axis (0 for 'sagittal', 1 for frontal and 2 for axial)

    # Returns
        3D numpy array, all selected slices concatenated.

    """
    for v in range(nifti_data.shape[0]):
        print('opening volume', str(v))
        vol_data=nifti_data[v,:,:,:]   ##choose volume v from nifti data=(200,176,208,176), vol_data=(176,208,176)
        vol_entropy=[]


        for s in range(vol_data.shape[Naxis]):
            if Naxis == 0:
                im=vol_data[s,:,:]    #take a sagittal slice image

            elif Naxis == 1:
                im=vol_data[:,s,:]    #take a coronal slice image

            elif Naxis == 2:
                im=vol_data[:,:,s]    #take an axial slice image
            ######image entropy normalized by mask area
            im_entropy=entropy(im)    ##compute its entropy
            vol_entropy.append(im_entropy)
        vol_entropy=np.array(vol_entropy)
        vol_entropy=vol_entropy[np.newaxis, :]    ###entropy array for all slices of each volume
        if v == 0:
            all_entropy=np.copy(vol_entropy)
        else:
            all_entropy=np.concatenate((all_entropy, vol_entropy))
    sorted_entropy=np.argsort(-all_entropy)     #sort slice's entropy in descending order
    slices=np.arange(Nslices)                   #select N slice indices with maximum entropy
    selected_indices=sorted_entropy[:,slices]
    print('selected indices:',selected_indices)
    return selected_indices


def resize_data(data, fx, fy):
    """ resizes 2D numpy array to a new dimension

    # Arguments
         data: 3D numpy array, shape=[array_length, h, w]
         fx: float, scaling factor along x axis
         fy: float, scaling factor along y axis

    #Returns
         3D numpy array with a new dimension.

    """
    k=0
    print('original data shape is:', data.shape)
    for i in range(data.shape[0]):
        if (i % 1000.0) == 0:
            k=k+1
            print('resizing images in progress:', k )

        im=data[i,:,:]
        resized_im=cv2.resize(im, None, fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)[np.newaxis,...]
        if i==0:
            new_data=np.copy(resized_im)
        else:
            new_data=np.concatenate((new_data, resized_im), axis=0)

    print('resized data shape is:', new_data.shape)
    return new_data


def generate_data(nifti_path, labels_path,  data_cv_pars, save_path, sampled_indices=None):
    """ generates a training data for the CNN.

    # Arguments
        nifti_path: 4D nifti file, MRI dataset.
        labels_path: path to csv file containing the labels of subjects.
        data_cv_pars: a dictionary with data preparation configuration.
        save_path: string path to save results.
        sampled_indices: 1d integer numpy array.

    # Returns
        sliced_data: float32 3d array data, shape=[n_images, im_size, im_size]
        labels: integer array, shape=[sliced_data.shape[0]]

    """
    start=time.time()
    Nslices=data_cv_pars['Nslices']
    dataset=data_cv_pars['dataset']
    labels_data=pd.read_csv(labels_path, delimiter=',')
    all_ids=labels_data['ID'].values.astype(np.int)
    labels=labels_data[data_cv_pars['label']].values
    if dataset == 'oasis_random':
        labels=np.random.choice([0,1], size=200)
    print('random labels:',labels)
    labels=np.repeat(labels, Nslices)
    labels=labels.astype(np.int)


    nifti_data=nib.load(nifti_path).get_data()  #numpy array of shape (176, 208, 176, 200)
    nifti_data=np.swapaxes(nifti_data, 0,3)
    nifti_data=np.swapaxes(nifti_data, 1,3)
    nifti_data=np.swapaxes(nifti_data, 2,3)    #after swapping axes, array shape (200,176, 208, 176)
    nifti_data=nifti_data[all_ids,:,:,:]

    slicing_plane=data_cv_pars['slicing_plane']
    print('slicing will be done along ', slicing_plane, ' plane')
    if slicing_plane == 'sagittal':
        Naxis=0
    elif slicing_plane == 'frontal':
        Naxis=1
    elif slicing_plane == 'axial':
        Naxis=2

    if data_cv_pars['data_size'] == 'some':

        print(nifti_data.shape)
        some_data=nifti_data[sampled_indices,:,:,:]
        labels=labels[sampled_indices]
        print(some_data.shape)
        nifti_data=some_data
#    labels=np.repeat(labels, Nslices)  ##########repeat the subject label to each slice


    '''choose_slice_indices(): computes entropy for each slice, and returns indices for N slices, Naxis=axial/sagittal/coronal'''
    selected_indices= choose_slice_indices(nifti_data, Naxis, Nslices)

    for v in range(nifti_data.shape[0]):
        print('volume number', v)
        indices=selected_indices[v,:]
        if Naxis == 0:
            current_data=nifti_data[v,indices,:,:]    #taking sagittal slices, current_data: images with shape (Nslices,176,208)
        elif Naxis == 1:
            current_data=nifti_data[v,:,indices,:]    #taking frontal slices, current_data: images with shape (Nslices,176,208)
        elif Naxis == 2:
            current_data=nifti_data[v,:,:,indices]    #taking axial slices, current_data: images with shape (Nslices,176,208)

        if v==0:
            sliced_data=np.copy(current_data)
        else:
            sliced_data=np.concatenate((sliced_data, current_data))

    prepared_data=sliced_data
    print('sliced array shape is:', prepared_data.shape)

    if data_cv_pars['data_resize']:
        resized_data=resize_data(sliced_data, fx=(224/prepared_data.shape[2]), fy=(224/prepared_data.shape[1]))
        prepared_data=resized_data
    if data_cv_pars['enable_hist_plot']:
        dataset_name=dataset
        plot_histogram(selected_indices, result_path, Nslices, dataset_name)

    end=time.time()
    print('Data processing done!')
    print('Data processing time in minutes:', (end - start) / 60 )

    processed_data=[prepared_data.astype(np.float32), labels]

    return processed_data  #processed data: (Nimages,176,208), for 5 slices: (1000,176,208)



def create_model(arch, learning_rate=0.00003, decay=0.1):
    """ creates a keras sequential model to be used with a KerasClassifier wrapper.

    # Arguments
        learning_rate: float, initialization of the learning rate for the optimizer.
        momentum: float, initialization of momentum for SGD optimizer.
        decay: float, initialization of the leaning rate decay.

    # Returns:
         model: keras sequential model

    """
    K.clear_session()

    if arch == 'resnet18':
        base_model=classification_models.resnet.ResNet18((224,224,3), weights='imagenet', include_top=False)
        out=GlobalAveragePooling2D()(base_model.output)
        out=Dense(1, activation='sigmoid', name='fc1')(out)
        model=Model(base_model.input, out)

    elif arch == 'vgg1':
        base_model=VGG16(weights='imagenet', include_top=False)
        model=None
        model=Sequential()
        for layer in base_model.layers[:-1]:
            model.add(layer)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(1, activation='sigmoid', name='d1'))
    elif arch == 'vgg2':
        base_model=VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
        model=Sequential()
        for layer in base_model.layers[:]:
            model.add(layer)
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(2, activation='softmax'))


    solver=Adam(lr=learning_rate, decay=decay)
    model.compile(optimizer=solver, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model




def generate_iterator(Nfolds, Nsbj, Nslices, cv_split, labels=None):
    """generate cross validation iterator.

    # Arguments
        Nfolds: integer, number of folds.
        Nsbj: integer, number of subjects.
        Nslices: integer, number of slices to choose.
        val_scheme: string, validation scheme ('holdout' for holdout method and 'cv' for cross validation).


    #Returns
        iterator: a list object, each list element is a tuple of train and test indices

    """

    if cv_split == 'correct':
        print('training on a dataset with subject-level split')
        all_ids=np.arange(int(Nsbj))
        perm=np.copy(all_ids)
        np.random.shuffle(perm)
        all_ids=all_ids[perm]            #####shuffling data

        kf=KFold(n_splits=Nfolds, shuffle=False)
        sbj_labels=None

        cv_iterator=[]
        for folds, indices in enumerate(kf.split(all_ids)):
            train_ids=all_ids[indices[0]]
            val_ids=all_ids[indices[1]]


            train_indices=vol2img_indices(train_ids, Nslices)
            val_indices=vol2img_indices(val_ids, Nslices)
            cv_iterator.append((train_indices, val_indices))

        iterator= cv_iterator
    elif cv_split == 'incorrect':
        print('training on a dataset with slice-level split')
        Nimages=int(Nsbj * Nslices)
        all_ids=np.arange(Nimages)
        perm=all_ids
        np.random.shuffle(perm)
        all_ids=all_ids[perm]

        kf=KFold(n_splits=Nfolds, shuffle=False)
        cv_iterator=[]
        for folds, indices in enumerate(kf.split(all_ids)):
            train_ids=all_ids[indices[0]]
            val_ids=all_ids[indices[1]]
            cv_iterator.append((train_ids, val_ids))
        iterator= cv_iterator

    return iterator




def train_on_array(nifti_path, arch_pars, data_cv_pars, result_path, sampled_indices=None, labels_path=None):
    """trains and evaluates the model.

    # Arguments
         nifti_path: 4D MRI nifti dataset.
         data_cv_pars: python dictionary, dictionary of configuration for data preparation.
         arch_pars: python dictionary, dictionary of configuration for model architecture.
         result_path: string, path to save the results.

    # Returns

    """
    batch_size=arch_pars['batch_size']
    epochs=arch_pars['epochs']

    Nslices=data_cv_pars['Nslices']
    Nfolds=data_cv_pars['Nfolds']
    cv_split=data_cv_pars['cv_split']
    print('data split method:', cv_split)

    all_data=generate_data(nifti_path, labels_path, data_cv_pars, result_path, sampled_indices)
    data, labels= all_data[0], all_data[1]
    Nsbj= data.shape[0] / Nslices

    print('number of subjects is:', Nsbj)
    print('number os slices:', Nslices)
    print('number of folds:', Nfolds)



    '''we generate training and validation indices for correct and incorrect cross validation
    cv is a list with length equal to the number of folds:
    Example: for a three fold cv: cv=((fold0_train, fold0_val), (fold1_train, fold1_val), (fold2_train, fold2_val))
    '''
    cv=generate_iterator(Nfolds, Nsbj, Nslices, cv_split, labels=labels)

    cv_method=data_cv_pars['cv_method']
    arch=arch_pars['pretrained_model']
    model=KerasClassifier(build_fn=create_model, arch=arch, epochs=epochs, batch_size=batch_size, verbose=1, class_weight='balanced')
    data=data[:,:,:,np.newaxis]
    data=(data - data.min()) * (255 / (data.max() - data.min() + K.epsilon()))
    data=single2three_channel(data)
    mean_data=np.mean(data)
    std_data=np.std(data)
    data=(data - mean_data) / std_data

    c_weights=class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    class_weights={0:c_weights[0], 1:c_weights[1]}

    if cv_method == "cross_validation":  ###cross_validate()
        scoring={'acc': 'accuracy',
                 'loss': 'neg_log_loss',
                 }

        cv_score=cross_validate(model, data, labels, cv=cv, scoring=scoring, return_train_score=True,verbose=1 )
        print('cross validation score is:', cv_score)

    elif cv_method == "grid_search":  ########GridSearchCV()
        scoring={'acc': 'accuracy'
                 }
        Ninner=data_cv_pars['Ninner']
        Nouter=data_cv_pars['Nouter']
        learning_rate=[ 0.00003, 0.0001, 0.0003, 0.001]#, 0.003, 0.01, 0.03]#, 0.00003, 0.0001, 0.0003, 0.001
        decay=[0, 0.1,0.3]
        epochs=[50,60]
        batch_size=[64,32]

        param_grid=dict(learning_rate=learning_rate,decay=decay)  #
        print(param_grid)


        Nsbj_inner=(Nsbj/Nouter)*(Nouter-1)
        outer_cv=generate_iterator(Nouter, Nsbj, Nslices, cv_split=cv_split, labels=labels)

        a=[]
        for i in range(len(outer_cv)):
            a.append(len(outer_cv[i][0]) / Nslices)

        print('number of subjects for each outer fold is:', a)

        if arch_pars['pretrained_model'] == 'vgg2':
            labels=to_categorical(labels)
            labels=labels.astype(np.float)

        inner_cv=generate_iterator(Ninner, Nsbj_inner, Nslices, cv_split=cv_split)

        clf=GridSearchCV(estimator=model,cv=inner_cv, param_grid=param_grid, scoring=scoring,
                         refit='acc', return_train_score=True, verbose=1)

        score=cross_validate(clf, data, y=labels, cv=outer_cv, scoring=scoring, return_train_score=True)
        print('the best scores on the best chosen parameters is:')
        print(score)




####################################################################################################################################
########################## Main function

def main(_):

    json.dump(arch_pars, open(os.path.join(result_path, 'arch_config.json'), 'w'))
    json.dump(data_cv_pars, open(os.path.join(result_path, 'data_config.json'), 'w'))
    start=time.time()

    ''' Random numpy array indices are generated to sample  the oasis dataset 10 times'''
    if data_cv_pars['data_size'] == 'subset':
        for i in range(10):
            con_indices=np.array(random.sample(range(100), int(data_cv_pars['sample_size'] / 2)))
            pos_indices=np.array(random.sample(range(100,200), int(data_cv_pars['sample_size'] / 2)))
            random_indices=np.concatenate((con_indices, pos_indices))
            print('indices for sample'+str(i), random_indices)

            r_path=os.path.join(result_path, 'sample'+str(i))
            if not os.path.isdir(r_path):
                os.mkdir(r_path)
            train_on_array(nifti_path, arch_pars, data_cv_pars, r_path,
                           sampled_indices=random_indices, labels_path=labels_path)
    else:
        train_on_array(nifti_path, arch_pars, data_cv_pars, result_path, labels_path=labels_path)
    end=time.time()
    print('fitting time in minutes:', (end-start) / 60)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('nifti_path', type=str, help='path to nifti file')
    parser.add_argument('labels_path', type=str, help='path to the labels file')
    parser.add_argument('arch_config_path', type=str, help='model config file path')
    parser.add_argument('data_CV_config_path', type=str, help='data config file path')
    parser.add_argument('--save_path', type=str, default='', help='location to save the results')



    #FLAGS = parser.parse_args()
    FLAGS =parser.parse_args()
    nifti_path=FLAGS.nifti_path
    arch_config_path=FLAGS.arch_config_path
    data_CV_config_path=FLAGS.data_CV_config_path
    labels_path=FLAGS.labels_path
    save_path=FLAGS.save_path
    print(labels_path)

    with open(arch_config_path, 'rb') as f:
        arch_pars=json.load(f)

    with open(data_CV_config_path, 'rb') as f:
        data_cv_pars=json.load(f)


    if save_path == '':
        path = os.path.realpath(sys.argv[0])
        if os.path.isdir(path):
            p=path
        else:
            p=os.path.dirname(path)

    else:
        p=save_path

    save_path=os.path.join(p, 'results')    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)


    dataset_path=os.path.join(save_path, data_cv_pars['dataset'])
    print(dataset_path)

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)



    now = datetime.datetime.now()
    current_datetime = [now.year, now.month, now.day, now.hour, now.minute]
    current_datetime = '_'.join(list(map(str, current_datetime)))

    result_path=os.path.join(dataset_path, str(data_cv_pars['Nslices'])+'slices'+current_datetime)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    main([sys.argv[0]] + sys.argv[1:])
