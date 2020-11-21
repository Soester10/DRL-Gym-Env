import random
import numpy as np
import tensorflow as tf
from collections import deque


class MountainCar_DQN():
    def __init__(self, max_len4train, UPDATE_SECONDARY_WEIGHTS, min_batch, min_len4train, DISCOUNT, Batch_Size):
        
        self.model = self.to_initialize_model()
        self.secondary_model = self.to_initialize_model()

##        self.model.load_weights("weights")

        self.secondary_model.set_weights(self.model.get_weights())

        self.model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        self.secondary_model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

        self.mem4train = deque(maxlen=max_len4train)

        self.UPDATE_SECONDARY_WEIGHTS = UPDATE_SECONDARY_WEIGHTS
        self.min_batch = min_batch
        self.min_len4train = min_len4train
        self.DISCOUNT = DISCOUNT
        self.Batch_Size = Batch_Size
            
    def to_initialize_model(self):
        
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, input_shape=(1,2)),
                                            tf.keras.layers.Dropout(0.1),
                                            tf.keras.layers.Dense(3, activation='linear')])

        return model

    def updating_mem4train(self,data):
        self.mem4train.append(data)                #[current_state, action, reward, new_current_state, done]

    def nn_training(self):
        if len(self.mem4train)<self.min_len4train:
            return

        training_data = random.sample(self.mem4train, self.min_batch)
        current_states = np.array([x[0].reshape(-1, *x[0].shape) for x in training_data])

        current_q_list = self.model.predict(current_states)

        new_current_states = np.array([x[3].reshape(-1, *x[3].shape) for x in training_data])
        
        future_q_list = self.secondary_model.predict(new_current_states)

        X=[]
        Y=[]

        for ind, training_data_iteration in enumerate(training_data):
            if not training_data_iteration[4]:
                max_future_q = np.max(future_q_list[ind])
                new_q = training_data_iteration[2] + self.DISCOUNT * max_future_q
            else:
                new_q = training_data_iteration[2]

            current_q = current_q_list[ind][0]
            current_q[training_data_iteration[1]] = new_q

            X.append([training_data_iteration[0]])
            Y.append(current_q)

        fiit = self.model.fit(np.array(X),np.array(Y), batch_size=self.Batch_Size, steps_per_epoch = len(np.array(X))/self.Batch_Size, verbose=0, epochs=4)


        if self.UPDATE_SECONDARY_WEIGHTS == True:
            self.secondary_model.set_weights(self.model.get_weights())
            self.UPDATE_SECONDARY_WEIGHTS = False
            

    def nn_predicting(self,cur_img):
        fp = cur_img.reshape(-1, *cur_img.shape)
        fp = fp.reshape(1,1,2)
        return self.model.predict(np.array(fp))[0]


