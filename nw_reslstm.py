# -*- coding: utf-8 -*
import os
import pickle
from keras.models import Model
from keras.layers.core import Dropout, Dense
from keras.layers import Input, merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from keras import backend as k
import numpy as np
import codecs
from keras.utils.visualize_util import plot

#### for devanagari text, reload system failsafe! ###
import sys
reload(sys)
sys.setdefaultencoding('utf8')
############################


np.random.seed(1707)

seq_length = 256  # seq size


##################### input text transform ##########################

def prepareData(path):
    # read input text
    with codecs.open(str(path), encoding='utf-8') as f:
        text = f.read()

    text = text.encode('utf-8')

    charset = sorted(set(text))  # sorted list of all characters in input

    num_char = int(len(charset))

    # map characters to index
    char_to_index = dict((c, i) for i, c in enumerate(charset))

    # map input characters to index
    input_to_index = [char_to_index[c] for c in text]

    # convert char_index to seq len and remove extra characters
    x = input_to_index[:len(input_to_index) - len(input_to_index) % seq_length]

    # convert char_index to matrix of dim -> ( seq no. , seq size)
    x = np.array(x, dtype='int32').reshape((-1, seq_length))

    # print  x
    # y zero matrix of 3d shape, dim -> (seq no. , seq size, len(charset))
    y = np.zeros((x.shape[0], x.shape[1], num_char), dtype='int32')

    # convert zero matrix to pos matrix with
    # 1 at char_index pos in 3rd dim and else 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            y[i, j, x[i, j]] = 1

    '''  
    Input:
        abc\n
        def

    seq_length = 3

    y= [[a b c]
        [\n d e]]

    y_roll=[[c a b]
            [e \n d]]     
        
    y_roll=[[\n a b]   
            [\n \n d]]
    '''

    # shift right 1
    y_roll = np.roll(y, 1, axis=1)
    # map 1st char of seq to \n
    y_roll[:, 0, :] = 0
    y_roll[:, 0, char_to_index['\n']] = 1

    return y_roll, y, num_char

########## Prepare data #############
print 'Preparing data'
X_train, y_train, train_char = prepareData('data/maha_lng.txt')


############ Model params ################
units = 256  # memory units
batch = 128  # batch_size
no_epochs = 59  # no. of epochs


############## Callbacks ###############

class decay_lr(Callback):
    '''
    n_epoch: no. of epochs after learning rate decays
    decay: value for decay
    '''

    def __init__(self, n_epoch, decay):
        super(decay_lr, self).__init__()
        self.n_epoch = n_epoch
        self.decay = decay

    def on_epoch_begin(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.get_value()
        if epoch > 1 and epoch % self.n_epoch == 0:
            new_lr = self.decay * old_lr
            k.set_value(self.model.optimizer.lr, new_lr)
        else:
            k.set_value(self.model.optimizer.lr, old_lr)

# decay learning rate
decay_sch = decay_lr(15, 0.5)

# checkpoint
checkpoint = ModelCheckpoint('weights/w_mal_reslstm_adam_noep{0}_batch{1}_seq_{2}.hdf5'.format(
    no_epochs, batch, seq_length), monitor='loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')

# early stopping
early_stopping = EarlyStopping(monitor='loss', mode='min', patience=5)



class lr_printer(Callback):
    '''
    Print learning rate at start of each epoch.
    '''
    def __init__(self):
        super(lr_printer, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        print('lr:', self.model.optimizer.lr.get_value())

# print learning rate
lr_print = lr_printer()


############## Build model ############

input = Input(shape=(seq_length, train_char), name='input')
x = LSTM(units, return_sequences=True, init='orthogonal', name='lstm1')(input)
x = Dropout(0.5, name='drop1')(x)
x = LSTM(units, return_sequences=True, init='orthogonal', name='lstm2')(x)
x = Dropout(0.5, name='drop2')(x)
x = TimeDistributed(Dense(train_char, init='orthogonal', name='fc1'))(x)
drop_op = Dropout(0.5, name='drop3')(x)
merge = merge([drop_op, input], mode='sum')
output = TimeDistributed(Dense(train_char, activation='softmax', init='orthogonal', name='fc2'))(merge)
model = Model(input=input, output=output)

model.load_weights("weights/weights_mal_nwreslstm5_adam_noep59_batch128_seq_256_loss0.566441147595.h5")


############## Optimizer ############
initlr = 7.81249946158e-06
adam = Adam(lr=initlr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=10)
print 'Compiling Model'
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
plot(model, to_file='nw_reslstm.png')


############ Fit model ############
history = History()
#print('Fitting model')
#model.fit(X_train, y_train, batch_size=batch, nb_epoch=no_epochs, validation_split=0.3,
#          callbacks=[history, early_stopping, checkpoint, decay_sch, lr_print])


######## Evaluate model ###############
'''print('Testing model')
score = model.evaluate(X_train, y_train, batch_size=batch)
print 'Loss:{}, Test accuracy:{}'.format(score[0], score[1])


########### Save weights #############
print 'Saving weights'
model.save_weights('weights/weights_noep{}_batch{}_seq_{}_loss{}.h5'.format(
    no_epochs, batch, seq_length, score[0]), overwrite=True)
#state = model.optimizer.get_state()
#pickle.dump(state, open('lstmres6_state_adam301.pkl', 'wb'))


############## Write history to file ########
with open('logs/nw_reslstm.txt', 'a') as log:
    log.write('\n\nloss:{0}\nacc:{1}\nval_loss:{2}\nval_acc:{3}\ntest loss:{4}\ntest accuracy:{5}\nlr:{6}'
              .format(history.history['loss'], history.history['acc'], history.history['val_loss'], history.history['val_acc'], score[0], score[1], model.optimizer.lr.get_value()))

'''
#os.system('shutdown now -h') # uncomment to shutdown system after training completion.


############ Generate text ############

with codecs.open(str('data/maha_lng.txt'), encoding='utf-8') as f:
    text = f.read()

text = text.encode('utf-8')
charset = sorted(set(text))
num_char = int(len(charset))
char_to_index = dict((c, i) for i, c in enumerate(charset))
index_to_char = dict((i, c) for i, c in enumerate(charset))

# Encodes char input to index
def encode_seed(seed, out):
    for i, c in enumerate(seed):
        out[0, i, char_to_index[c]] = 1

# Sampling function
def sample(a, temperature):
    a = np.asarray(a).astype('float64')
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# Predict char and feed it to the model to predict next char in seq
def generate_text(model, diversity, seed, seq_count=seq_length):
    out = seed
    print 'seed:', seed, ' diversity:', diversity
    # zero vector of shape (1, seq_length, train_chars)
    hot_vec = np.zeros((1, seq_count, num_char), dtype='float64')
    # one hot vec of seed
    encode_seed(out, hot_vec)
    # iterate over seq.
    # print hot_vec.shape
    while len(out) < seq_count:
        # make probability distribution (prob dist) over predicted characters
        prob_dist = model.predict({str(model.input.name): hot_vec}, verbose=0)
        # sample char index using prob dist
        prob_dist = prob_dist[0][len(out) - 1]
        char = index_to_char[sample(prob_dist, diversity)]
        # add char to seed
        out += char
        # encode one hot_vec with gnerated char
        encode_seed(out, hot_vec)
    return out

# Take seed value from user
seed = ['संजय', 'धृतराष्ट्']


print 'Generating chars'
for i in seed:
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print(generate_text(model, diversity, i))


