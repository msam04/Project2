
# coding: utf-8

# In[ ]:


from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

cifar10_dataset_folder_path= 'cifar-10-batches-py'

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10 #data belongs to classes {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
n_inputs = 3072 #image size is 32X32 and 3 channels - 
batch_size = 50

class DLProgress(tqdm):
    last_block= 0
    
    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total= total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block= block_num
        
        
if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz','cifar-10-python.tar.gz',pbar.hook)
        
if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        tar.extractall()
        tar.close()
        
def _load_label_names():
    """
    Load the label names from file
    """
    
    return['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    """
    Load a batch of the dataset
    """
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']

    return features, labels

def display_stats(cifar10_dataset_folder_path, batch_id, sample_id):
    """
    Display Stats of the the dataset
    """
    batch_ids = list(range(1, 6))

    if batch_id not in batch_ids:
        print('Batch Id out of Range. Possible Batch Ids: {}'.format(batch_ids))
        return None

    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)

    if not (0 <= sample_id < len(features)):
        print('{} samples in batch {}.  {} is out of range.'.format(len(features), batch_id, sample_id))
        return None

    print('\nStats of batch {}:'.format(batch_id))
    print('Samples: {}'.format(len(features)))
    print('Label Counts: {}'.format(dict(zip(*np.unique(labels, return_counts=True)))))
    print('First 20 Labels: {}'.format(labels[:20]))

    sample_image = features[sample_id]
    sample_label = labels[sample_id]
    label_names = _load_label_names()

    print('\nExample of Image {}:'.format(sample_id))
    print('Image - Min Value: {} Max Value: {}'.format(sample_image.min(), sample_image.max()))
    print('Image - Shape: {}'.format(sample_image.shape))
    print('Label - Label Id: {} Name: {}'.format(sample_label, label_names[sample_label]))
    plt.axis('off')
    plt.imshow(sample_image)
    #plt.show(sample_image)
    plt.show()
    
def normalize(list_image_data):
    for i in range(len(list_image_data)):
        list_image_data[i] = list_image_data[i]/np.mean(list_image_data[i])
    return list_image_data 

def one_hot_encode(label_value_list):
    #all_labels = _load_label_names()
    coded_label_list = []
    for i in range(len(label_value_list)):
        coded_label = [0]*n_classes
        #ind = all_labels.index(label_value)
        #coded_label[ind] = 1
        coded_label[label_value_list[i] - 1] = 1
        coded_label_list.append(coded_label)
    return coded_label_list


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    """
    Preprocess data and save it to file
    """
    
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    """
    Preprocess Training and Validation Data
    """
    n_batches = 5
    valid_features = []
    valid_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)
        validation_count = int(len(features) * 0.1)
        #print("validation count: ",validation_count)
        
        #print(features.shape)
        
        
        # Prprocess and save a batch of training data
        _preprocess_and_save(
            normalize,
            one_hot_encode,
            features[:-validation_count],
            labels[:-validation_count],
            'preprocess_batch_' + str(batch_i) + '.p')

        # Use a portion of training batch for validation
        valid_features.extend(features[-validation_count:])
        valid_labels.extend(labels[-validation_count:])

    # Preprocess and Save all validation data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(valid_features),
        np.array(valid_labels),
        'preprocess_validation.p')

    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    # load the training data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    # Preprocess and Save all training data
    _preprocess_and_save(
        normalize,
        one_hot_encode,
        np.array(test_features),
        np.array(test_labels),
        'preprocess_training.p')


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        #print(features[start].shape)
        yield features[start:end], labels[start:end]

def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)

def load_all_validation_data(batch_size):
    filename = 'preprocess_validation.p'
    if(isfile(filename)):
        features, labels = pickle.load(open(filename, mode='rb'))
        return batch_features_labels(features, labels, batch_size)
    
def load_all_training_data(batch_size):
    filename = 'preprocess_training.p'
    if(isfile(filename)):
        features, labels = pickle.load(open(filename, mode='rb'))
        return batch_features_labels(features, labels, batch_size)
    else:
        exit(-1)

    

def load_training_batch(batch_size, batch_number):
    filename = cifar10_dataset_folder_path + '/test_batch/'
    filename = 'preprocess_training.p'
    if(isfile(filename)):
        features, labels = pickle.load(open(filename, mode='rb'))
        return batch_features_labels(features, labels, batch_size)
    else:
        exit(-1)
    

def load_validation_batch(batch_size, batch_number):
    #filename = cifar10_dataset_folder_path + '/test_batch/'
    filename = 'preprocess_validation.p'
    if(isfile(filename)):
        features, labels = pickle.load(open(filename, mode='rb'))
        return batch_features_labels(features, labels, batch_size)
    else:
        exit(-1)
    


def display_image_predictions(features, labels, predictions):
    n_classes = 10
    label_names = _load_label_names()
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    label_ids = label_binarizer.inverse_transform(np.array(labels))

    fig, axies = plt.subplots(nrows=4, ncols=2)
    fig.tight_layout()
    fig.suptitle('Softmax Predictions', fontsize=20, y=1.1)

    n_predictions = 3
    margin = 0.05
    ind = np.arange(n_predictions)
    width = (1. - 2. * margin) / n_predictions

    for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
        pred_names = [label_names[pred_i] for pred_i in pred_indicies]
        correct_name = label_names[label_id]

        axies[image_i][0].imshow(feature*255)
        axies[image_i][0].set_title(correct_name)
        axies[image_i][0].set_axis_off()

        axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
        axies[image_i][1].set_yticks(ind + margin)
        axies[image_i][1].set_yticklabels(pred_names[::-1])
        axies[image_i][1].set_xticks([0, 0.5, 1.0])
        


## x is your data, you do not need to specify the size of the matrix [None, n_inputs], 
## but this will cause tensorflow to throw an error if data is loaded outside of that shape
## This may need to be changed if your data set is not the mnist
x = tf.placeholder('float',[None, n_inputs])
y = tf.placeholder('float',[None, n_classes])
#x is a placeholder, a value we'll input when we ask tensorflow to run a computation.
#The first argument which is 'None' means that a dimension can be of any length. 
W = tf.Variable(tf.zeros([n_inputs,n_classes]))
b = tf.Variable(tf.zeros([n_classes]))

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
#Outputs random values from a truncated normal distribution.
#The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#The strides argument specifies how much the window shifts in each of the input dimensions. The dimensions are batch, height, width, chanels. Here, because the image is greyscale, there is only 1 dimension. A batch size of 1 means that a single example is processed at a time. A value of 1 in row/column means that we apply the operator at every row and column. When the padding parameter takes the value of 'SAME', this means that for those elements which have the filter sticking out during convolution, the portions of the filter that stick out are assumed to contribute 0 to the dot product. The size of the image remains the same after the operation remains the same. When the padding = 'VALID' option is used, the filter is centered along the center of the image, so that the filter does not stick out. So the output image size would have been 3X3. 

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def neural_network_model(data):
    W_fc1 = weight_variable([n_inputs, n_nodes_hl1])
    b_fc1 = bias_variable([n_nodes_hl1])

    W_fc2 = weight_variable([n_nodes_hl1, n_nodes_hl2])
    b_fc2 = bias_variable([n_nodes_hl2])

    W_fc3 = weight_variable([n_nodes_hl2, n_nodes_hl3])
    b_fc3 = bias_variable([n_nodes_hl3])

    W_output = weight_variable([n_nodes_hl3,n_classes])
    b_output = bias_variable([n_classes])

    l1 = tf.nn.relu(tf.matmul(data,W_fc1) + b_fc1)
    l2 = tf.nn.relu(tf.matmul(l1,W_fc2) + b_fc2)
    l3 = tf.nn.relu(tf.matmul(l2,W_fc3) + b_fc3)

    output = tf.matmul(l3, W_output) + b_output
    return output

def conv_network_model(x):
        
    W_conv1 = weight_variable([5, 5, 3, 32])
    #Weight variable dimensions - filter dimensions - width x height x channels x number of filters
    #Number of channels in output image - number of filters in the filter
    b_conv1 = bias_variable([32])

    
    x_image = tf.reshape(x, [-1,32,32,3])
    x_image = tf.cast(x_image, tf.float32)
#A value of -1 for the reshape parameter is to allow the input to dynamically decide what value that parameter should take.
#Since we use x_image as an input to conv2d, it would seem we are changing it from the 1X784 shape to a shape which has 4 parameters. 
#The first new parameter would be the number of images. 
#The second and third would be the dimensions of the image.
#The fourth would be the number of channels in the input image. 


    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#After convolution, the number of images would be the same as that in x_image. The dimensions would depend on image size, filter size and padding. The number of channels would depend on the number of channels in the filter.
    h_pool1 = max_pool_2x2(h_conv1)
#At the end of the first pooling layer, there is a 14X14 image.

    #print("Layer 1")

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
#At the end of the second pooling layer, there is a 8X8 image.
    #print("Layer 2")

    W_fc1 = weight_variable([8 * 8 * 64, 1024])
    b_fc1 = bias_variable([1024])
#1024 - an arbitrary choice?

    #print(h_pool2.shape)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#    keep_prob = tf.placeholder(tf.float32)
#    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

#    output=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    output=tf.matmul(h_fc1, W_fc2) + b_fc2
    
       
    return output


def train_neural_network(x, network_model):
    prediction = network_model(x)
    #print(prediction[0])
#    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
#logit - unnormalized log of probability
#softmax - allows normalization of values
    #train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
    train_step = tf.train.GradientDescentOptimizer(1e-2).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#tf.equal - returns the truth value of whether the 2 arguments were equal.
#checks whether for the first dimension the largest probabiliry with one hot encoding is the same with the predicted and actual values. 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#reduce_mean reduces all dimensions by computing the mean.
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(100):
            count = 0
            for t_b in load_all_training_data(25):
                count += 1
                flattened_x = np.reshape(t_b[0], [-1, 32*32*3])
                ##prediction = network_model(flattened_x)
                
                train_step.run(feed_dict={x: flattened_x, y: t_b[1]})
                train_accuracy = sess.run(accuracy, feed_dict={x:flattened_x, y:t_b[1]})
                if(count % 100 == 0):
                    print("for training batch: {}, iteration: {}, accuracy: {}".format(count,i,train_accuracy))
                    
            v_count = 0   
            v_accuracy = 0
            for v_b in load_all_validation_data(25):
                v_count += 1
                flattened_v_x = np.reshape(v_b[0], [-1, 32*32*3])
                #print("test accuracy %g"%accuracy.eval(feed_dict={x: flattened_v_x, y: v_b[1]})) 
                test_accuracy = sess.run(accuracy, feed_dict={x: flattened_v_x, y: v_b[1]})
                if(v_count % 50 == 0):
                    print("for validation batch: {}, iteration: {}, accuracy: {}".format(v_count,i,test_accuracy))
                v_accuracy += test_accuracy
                average_v_accuracy = v_accuracy / v_count
                
                
            print("For run {}, test accuracy is: {}".format(i, average_v_accuracy))
            if(average_v_accuracy >= 0.75):
                print("Stopping at accuracy: {}, iteration: {}".format(average_v_accuracy, i))
                
                
        #test_data_v = load_all_validation_data()
        #flattened_test_data_x = np.reshape(test_data_v[0], [-1, 32*32*3])
        #print("testall accuracy %g"%accuracy.eval(feed_dict={x: flattened_test_data_x, y: test_data_v[1]})) 
        
                
#train_neural_network(x, neural_network_model)
# Explore the dataset
#batch_id= 1
#sample_id= 5
#display_stats(cifar10_dataset_folder_path, batch_id, sample_id)  
#image_data, image_labels = load_cfar10_batch(cifar10_dataset_folder_path,1)
preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
train_neural_network(x, conv_network_model)
print("done")



