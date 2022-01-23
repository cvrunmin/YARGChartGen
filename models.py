import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import layers
from tensorflow.keras import models

import utils
from utils import get_mel_spectrogram

class OnsetModel:
    '''
    an onset model defined in the following paper:

    Y. Liang, W. Li, Kokolo Ikeda. Procedural Content Generation of Rhythm Games Using Deep Learning Methods.
    https://doi.org/10.1007/978-3-030-34644-7_11

    C. Donahue, Z.C. Lipton, J. McAuley. Dance dance convolution. available http://proceedings.mlr.press/v70/donahue17a.html
    '''
    def __init__(self,
                lstm_drop_rate=0.5,
                full_connection_layer_drop_rate=0.5,
                *,
                load_file_path=None):
        if not load_file_path is None:
            # self.model = tf.saved_model.load(load_file_path)
            self.model = models.load_model(load_file_path)
        else:
            frq_input = tf.keras.Input(shape=(15,80,3), name='frq')
            difficulty_input = tf.keras.Input(shape=(5,), name='difficulty')
            conv1 = layers.Conv2D(10, (7,3), activation='relu')(frq_input)
            pool1 = layers.MaxPool2D(pool_size=(1,3))(conv1)
            conv2 = layers.Conv2D(20, 3, activation='relu')(pool1)
            pool2 = layers.MaxPool2D(pool_size=(1,3))(conv2)
            flattener = layers.Flatten()(pool2)
            # flattener = layers.Reshape(target_shape=(7, 8 * 20))(pool2)
            lvl_concat = layers.Concatenate()([flattener, difficulty_input])
            reshaper = layers.Reshape((1,-1))(lvl_concat)
            lstm = layers.Bidirectional(layers.LSTM(200))(reshaper)
            if lstm_drop_rate < 1.0:
                lstm = layers.Dropout(lstm_drop_rate)(lstm)
            pass1 = layers.Dense(256, activation='relu')(lstm)
            if full_connection_layer_drop_rate < 1.0:
                pass1 = layers.Dropout(full_connection_layer_drop_rate)(pass1)
            pass2 = layers.Dense(128, activation='relu')(pass1)
            if full_connection_layer_drop_rate < 1.0:
                pass2 = layers.Dropout(full_connection_layer_drop_rate)(pass2)
            prob_onset = layers.Dense(1, activation='sigmoid', name='steps')(pass2)

            self.model = tf.keras.Model(inputs=[frq_input, difficulty_input], outputs=prob_onset)
            self.model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['binary_accuracy', 'binary_crossentropy'])
        self.history = None

    
    def model_summary(self):
        self.model.summary()

    
    def save(self, file_path):
        tf.saved_model.save(self.model, file_path)
    

    def train(self, train_ds, epochs=5, val_ds=None):
        self.history = self.model.fit(x=train_ds, epochs=epochs, validation_data=val_ds)



    def query(self, audio_file=None, audio_lmel80=None, difficulty=0, threshold=0.5, should_smooth=True, smooth_length=7):
        '''query a set of note event positions from an audio data
        
        audio_file: the file path of the audio file
        audio_lmel80: the 80-bin log mel-scale spectrogram data. Should stack with fft_length 1024, 2048, 4096
        difficulty: an integer value in [0,5). 0 is the easiest, and 4 is the most difficult
        threshold: a threshold value to determine whether the frame should have note event or not based on 
        predicted probability.
        should_smooth: whether predicted sequence should be smoothen by hamming window or not. default True
        smooth_length: if should_smooth, the length of the hamming window.

        return a list of frame position
        '''
        from scipy.signal import find_peaks
        if audio_file is None and audio_lmel80 is None:
            raise ValueError("either audio_file or audio_lmel80 must have value")
        if difficulty < 0 or difficulty > 4:
            raise ValueError(f"difficulty out of range: expected: [0,5)  received:{difficulty}")
        if audio_lmel80 is None:
            audio = tfio.audio.AudioIOTensor(audio_file)
            audio_sample = tf.cast(tf.math.reduce_mean(audio.to_tensor(), axis=1), tf.float32)
            fs = audio.rate.numpy()
            frame_step = fs // 100
            audio_lmel80 = tf.stack([get_mel_spectrogram(audio_sample, 1024, frame_step, fs),
                    get_mel_spectrogram(audio_sample, 2048, frame_step, fs),
                    get_mel_spectrogram(audio_sample, 4096, frame_step, fs)], axis=2)
        padded = tf.concat([tf.zeros((7,80,3)), audio_lmel80,tf.zeros((7,80,3))],axis=0)
        frame_frq = tf.signal.frame(padded, 15, 1, axis=0)
        prediction = self.model.predict([frame_frq, tf.repeat(tf.expand_dims(tf.one_hot(difficulty, 5),axis=0), tf.shape(frame_frq)[0], axis=0)])
        if should_smooth:
            prediction = tf.squeeze(tf.nn.conv1d(prediction.reshape([1,-1,1]), tf.reshape(tf.signal.hamming_window(smooth_length, periodic=False), [-1,1,1]), stride=1, padding='SAME'))
        peaks, _ = find_peaks(prediction, height=threshold)
        return peaks


class NoteActionModel:

    time_interval_bar = np.array([5,10,20,40,80,160,320])
    normal_type_mask = tf.constant([0,0,0,0,-float('inf'), -float('inf')])
    sliding_type_mask = tf.constant([0,-float('inf'),-float('inf'),-float('inf'), 0, 0])

    left_keep_mask :tf.Tensor
    right_keep_mask :tf.Tensor

    def __init__(self,
                lstm_drop_rate=0.5,
                lstm_cell=256,
                *,
                load_file_path=None):
        mask_list = tf.constant([utils.flat_pos_width(x,w) for x in range(0, 12) for w in range(1, 13 - x) if x + w - 1 >= 6], dtype=tf.int64)
        self.left_keep_mask = tf.sparse.to_dense(tf.sparse.SparseTensor(tf.reshape(mask_list, (-1, 1)), [-float('inf')]*len(mask_list), (78,)))
        mask_list = tf.constant([utils.flat_pos_width(x,w) for x in range(0, 6) for w in range(1, 13 - x)], dtype=tf.int64)
        self.right_keep_mask = tf.sparse.to_dense(tf.sparse.SparseTensor(tf.reshape(mask_list, (-1, 1)), [-float('inf')]*len(mask_list), (78,)))
        lstm_config_1, lstm_config_2 = None, None
        if not load_file_path is None:
            # self.model = tf.saved_model.load(load_file_path)
            self.model = models.load_model(load_file_path)
            configs = [layer.get_config() for layer in self.model.layers if isinstance(layer, layers.LSTM)]
            for config in configs:
                if 'return_sequences' in config and config['return_sequences']:
                    lstm_config_1 = config.copy()
                else:
                    lstm_config_2 = config.copy()
        else:
            frq_input = tf.keras.Input(shape=(15,80,3), name='frq')
            difficulty_input = tf.keras.Input(shape=(5,), name='difficulty')
            note_type_input_1 = tf.keras.Input(shape=(6,), name='note_type_1')
            note_xw_input_1 = tf.keras.Input(shape=(78,), name='note_xw_1')
            note_type_input_2 = tf.keras.Input(shape=(6,), name='note_type_2')
            note_xw_input_2 = tf.keras.Input(shape=(78,), name='note_xw_2')
            time_interval_input = tf.keras.Input(shape=(8,), name='time_interval_before')
            time_interval_input_2 = tf.keras.Input(shape=(8,), name='time_interval_after')
            conv1 = layers.Conv2D(10, (7,3), activation='relu')(frq_input)
            pool1 = layers.MaxPool2D(pool_size=(1,3))(conv1)
            conv2 = layers.Conv2D(20, 3, activation='relu')(pool1)
            pool2 = layers.MaxPool2D(pool_size=(1,3))(conv2)
            flattener = layers.Flatten()(pool2)
            # flattener = layers.Reshape(target_shape=(7, 8 * 20))(pool2)
            lvl_concat = layers.Concatenate()([flattener, difficulty_input, note_type_input_1, note_xw_input_1, note_type_input_2, note_xw_input_2,time_interval_input,time_interval_input_2])
            reshaper = layers.Reshape((1,-1))(lvl_concat)
            lstm1 = layers.LSTM(lstm_cell, return_sequences=True)(reshaper)
            if lstm_drop_rate < 1.0:
                lstm1 = layers.Dropout(lstm_drop_rate)(lstm1)
            lstm2 = layers.LSTM(lstm_cell)(lstm1)
            if lstm_drop_rate < 1.0:
                lstm2 = layers.Dropout(lstm_drop_rate)(lstm2)
            # pass1 = layers.Dense(256, activation='relu')(lstm2)
            # full_connection_layer_drop_rate = 0.5
            # if full_connection_layer_drop_rate < 1.0:
            #     pass1 = layers.Dropout(full_connection_layer_drop_rate)(pass1)
            # lstm2 = pass1
            # dropout = layers.Dropout(0.5)(lstm)
            note_type_output_1 = layers.Dense(6, name='note_type_out_1')(lstm2)
            note_xw_output_1 = layers.Dense(78, name='note_xw_out_1')(lstm2)
            note_type_output_2 = layers.Dense(6, name='note_type_out_2')(lstm2)
            note_xw_output_2 = layers.Dense(78, name='note_xw_out_2')(lstm2)

            self.model = tf.keras.Model(inputs=[frq_input, difficulty_input,note_type_input_1,note_xw_input_1,note_type_input_2,note_xw_input_2,time_interval_input,time_interval_input_2],
                                outputs=[note_type_output_1,note_xw_output_1,note_type_output_2,note_xw_output_2])
            self.model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['categorical_accuracy'])
        # inference part
        frq_input = tf.keras.Input(shape=(15,80,3), batch_size=1, name='frq')
        difficulty_input = tf.keras.Input(shape=(5,), batch_size=1, name='difficulty')
        note_type_input_1 = tf.keras.Input(shape=(6,), batch_size=1, name='note_type_1')
        note_xw_input_1 = tf.keras.Input(shape=(78,), batch_size=1, name='note_xw_1')
        note_type_input_2 = tf.keras.Input(shape=(6,), batch_size=1, name='note_type_2')
        note_xw_input_2 = tf.keras.Input(shape=(78,), batch_size=1, name='note_xw_2')
        time_interval_input = tf.keras.Input(shape=(8,), batch_size=1, name='time_interval_before')
        time_interval_input_2 = tf.keras.Input(shape=(8,), batch_size=1, name='time_interval_after')
        conv1 = layers.Conv2D(10, (7,3), activation='relu')(frq_input)
        pool1 = layers.MaxPool2D(pool_size=(1,3))(conv1)
        conv2 = layers.Conv2D(20, 3, activation='relu')(pool1)
        pool2 = layers.MaxPool2D(pool_size=(1,3))(conv2)
        flattener = layers.Flatten()(pool2)
        lvl_concat = layers.Concatenate()([flattener, difficulty_input, note_type_input_1, note_xw_input_1, note_type_input_2, note_xw_input_2,time_interval_input,time_interval_input_2])
        reshaper = layers.Reshape((1,-1))(lvl_concat)
        if not (lstm_config_1 is None):
            lstm_config_1['stateful'] = True
            lstm1_layer = layers.LSTM.from_config(lstm_config_1)
        else:
            lstm1_layer = layers.LSTM(lstm_cell, return_sequences=True, stateful=True)
        lstm1 = lstm1_layer(reshaper)
        if lstm_drop_rate < 1.0:
            lstm1 = layers.Dropout(lstm_drop_rate)(lstm1)
        if not (lstm_config_2 is None):
            lstm_config_2['stateful'] = True
            lstm2_layer = layers.LSTM.from_config(lstm_config_2)
        else:
            lstm2_layer = layers.LSTM(lstm_cell, stateful=True)
        lstm2 = lstm2_layer(lstm1)
        if lstm_drop_rate < 1.0:
            lstm2 = layers.Dropout(lstm_drop_rate)(lstm2)
        # pass1 = layers.Dense(256, activation='relu')(lstm2)
        # full_connection_layer_drop_rate = 0.5
        # if full_connection_layer_drop_rate < 1.0:
        #     pass1 = layers.Dropout(full_connection_layer_drop_rate)(pass1)
        # lstm2 = pass1
        # dropout = layers.Dropout(0.5)(lstm)
        note_type_output_1 = layers.Dense(6, name='note_type_out_1')(lstm2)
        note_xw_output_1 = layers.Dense(78, name='note_xw_out_1')(lstm2)
        note_type_output_2 = layers.Dense(6, name='note_type_out_2')(lstm2)
        note_xw_output_2 = layers.Dense(78, name='note_xw_out_2')(lstm2)

        self.inference_model = tf.keras.Model(inputs=[frq_input, difficulty_input,note_type_input_1,note_xw_input_1,note_type_input_2,note_xw_input_2,time_interval_input,time_interval_input_2],
                            outputs=[note_type_output_1,note_xw_output_1,note_type_output_2,note_xw_output_2])
        self.history = None
        if not load_file_path is None:
            self.transfer_train_param_to_inference()

    def model_summary(self):
        self.model.summary()

    
    def save(self, file_path):
        tf.saved_model.save(self.model, file_path)

    
    def save_inference(self, file_path):
        tf.saved_model.save(self.inference_model, file_path)


    def transfer_train_param_to_inference(self):
        for i in range(len(self.model.layers)):
            self.inference_model.layers[i].set_weights(self.model.layers[i].get_weights())

    
    def reset_inference_lstm_state(self):
        self.inference_model.reset_states()
    

    def train(self, train_ds, epochs=5, val_ds=None):
        self.history = self.model.fit(x=train_ds, epochs=epochs, validation_data=val_ds)

    def query(self, audio_file=None, audio_lmel80=None, difficulty=0, steps=np.empty((0,)), *, temperature=0.5, seeds:list=None):
        '''query a set of note event positions from an audio data
        
        audio_file: the file path of the audio file
        audio_lmel80: the 80-bin log mel-scale spectrogram data. Should stack with fft_length 1024, 2048, 4096
        difficulty: an integer value in [0,5). 0 is the easiest, and 4 is the most difficult
        steps: key frames that should have note action

        return a list of note action
        '''
        if audio_file is None and audio_lmel80 is None:
            raise ValueError("either audio_file or audio_lmel80 must have value")
        if difficulty < 0 or difficulty > 4:
            raise ValueError(f"difficulty out of range: expected: [0,5)  received:{difficulty}")
        if audio_lmel80 is None:
            audio = tfio.audio.AudioIOTensor(audio_file)
            audio_sample = tf.cast(tf.math.reduce_mean(audio.to_tensor(), axis=1), tf.float32)
            fs = audio.rate.numpy()
            frame_step = fs // 100
            audio_lmel80 = tf.stack([get_mel_spectrogram(audio_sample, 1024, frame_step, fs),
                    get_mel_spectrogram(audio_sample, 2048, frame_step, fs),
                    get_mel_spectrogram(audio_sample, 4096, frame_step, fs)], axis=2)
        assert temperature > 0
        self.reset_inference_lstm_state()
        padded = tf.concat([tf.zeros((7,80,3)), audio_lmel80,tf.zeros((7,80,3))],axis=0)
        diff_ts = tf.one_hot(difficulty, 5)
        if isinstance(seeds, list):
            seed_type_1, seed_xw_1, seed_type_2, seed_xw_2 = seeds[0], seeds[1], seeds[2], seeds[3]
        else:
            seed_type_1 = tf.one_hot(0,6)
            seed_xw_1 = tf.one_hot(0,78)
            seed_type_2 = tf.one_hot(0,6)
            seed_xw_2 = tf.one_hot(0,78)
        note_type_1, note_xw_1, note_type_2, note_xw_2 = [],[],[],[]
        step_delta = steps[1:] - steps[:-1]
        step_delta = np.digitize(step_delta, self.time_interval_bar)
        backward_time = np.concatenate(([7], step_delta))
        forward_time = np.concatenate((step_delta,[7]))
        i = 0
        c0_long_press, c1_long_press = False, False
        for frame in steps:
            print(f'\r Progress {i}/{len(steps)}, frame:{frame}', end = '\r')
            if frame >= tf.shape(audio_lmel80)[0]:
                win_frame = tf.zeros((15,80,3))
            else:
                win_frame = padded[frame:frame+15]
            predicts = self.inference_model.predict([tf.expand_dims(win_frame, 0), tf.expand_dims(diff_ts, 0), tf.expand_dims(seed_type_1, 0), tf.expand_dims(seed_xw_1, 0), tf.expand_dims(seed_type_2, 0), tf.expand_dims(seed_xw_2, 0), tf.expand_dims(tf.one_hot(backward_time[i], 8), 0),tf.expand_dims(tf.one_hot(forward_time[i], 8), 0)])
            predicts[0] = np.array(predicts[0]) / temperature
            predicts[1] = np.array(predicts[1]) / temperature
            predicts[2] = np.array(predicts[2]) / temperature
            predicts[3] = np.array(predicts[3]) / temperature
            if c0_long_press:
                predicts[0] = predicts[0] + self.sliding_type_mask
                predicts[1] = predicts[1] + self.left_keep_mask
            else:
                predicts[0] = predicts[0] + self.normal_type_mask
                if c1_long_press:
                    predicts[1] = predicts[1] + self.left_keep_mask
            if c1_long_press:
                predicts[2] = predicts[2] + self.sliding_type_mask
                predicts[3] = predicts[3] + self.right_keep_mask
            else:
                predicts[2] = predicts[2] + self.normal_type_mask
                if c0_long_press:
                    predicts[3] = predicts[3] + self.right_keep_mask
            type1 = tf.squeeze(tf.random.categorical(predicts[0], num_samples=1)).numpy()
            type2 = tf.squeeze(tf.random.categorical(predicts[2], num_samples=1)).numpy()
            if not c0_long_press and type1 == 3:
                predicts[1] = predicts[1] + self.left_keep_mask
                if type2 < 3:
                    predicts[3] = predicts[3] + self.right_keep_mask
            if not c1_long_press and type2 == 3:
                predicts[3] = predicts[3] + self.right_keep_mask
                if type1 < 3:
                    predicts[1] = predicts[1] + self.left_keep_mask
            xw1 = tf.squeeze(tf.random.categorical(predicts[1], num_samples=1)).numpy()
            xw2 = tf.squeeze(tf.random.categorical(predicts[3], num_samples=1)).numpy()
            if type1 == 3:
                c0_long_press = True
            elif type1 == 5:
                c0_long_press = False
            if type2 == 3:
                c1_long_press = True
            elif type2 == 5:
                c1_long_press = False
            x1, w1 = utils.get_pos_width_from_flatten(xw1)
            x2, w2 = utils.get_pos_width_from_flatten(xw2)
            if (type1 != 0 and type2 != 0) and (x1 + w1 - 1 >= x2):
                if x2 + w2 - x1 < 4:
                    # try merging notes
                    if (type1 >= 3 or (x2 + w2 + x1) / 2 < 5.5)  and type2 < 3:
                        type2 = 0
                        w1 = x2 + w2 - x1
                    elif (type2 >= 3 or (x2 + w2 + x1) / 2 >= 5.5) and type1 < 3:
                        type1 = 0
                        w2 = x2 + w2 - x1
                        x2 = x1
                    else:
                        if x1 < 3:
                            x2 = x1 + w1
                        elif x2 + w2 - 1 > 8:
                            x1 = x2 - w1
                        else:
                            # force centering notes
                            x1 = 6 - w1
                            x2 = 6
                elif x2 >= x1 and x1 + w1 <= x2 + w2 or x1 >= x2 and x2 + w2 <= x1 + w1:
                    # one note is inside other note
                    if type1 < 3:
                        type1 = 0
                    elif type2 < 3:
                        type2 = 0
                    else:
                        x1 = x2
                        w1 = w2 // 2
                        tmp = x2 + w2
                        w2 = w2 // 2
                        x2 = tmp - w2
                elif x1 == 0 and w1 < 3:
                    # push away note 1
                    tmp1 = min(x2 + w2 - 1, 11)
                    x2 = max(x1 + w1 - 1, x2 + 1)
                    w2 = tmp1 - x2 + 1
                elif x2 + w2 - 1 == 11 and w2 < 3:
                    # push away note 0
                    tmp1 = min(x1 + w1 - 1, x2 - 1)
                    x1 = min(x1, tmp1)
                    w1 = tmp1 - x1 + 1
                elif x1 + w1 - 1 == x2:
                    # reduce width by 1
                    if w1 < w2:
                        w2 -= 1
                    else:
                        w1 -= 1
                elif (x2 + w2 - 1) - (x1 + w1 - 1) <= 2 or x2 - x1 <= 2:
                    # take average of widths
                    tmp = (x2 + w2 - 1) - x1 + 1
                    w1 = tmp // 2
                    x2 = (x2+w2) - tmp // 2
                    w2 = tmp // 2
                else:
                    # switch right pos of note 0 and left pos of note 1
                    tmp = w1
                    tmp2 = x2 + w2 - 1
                    w1 = x2 - x1 + 1
                    x2 = x1 + tmp - 1
                    w2 = tmp2 - x2 + 1
            xw1 = utils.flat_pos_width(x1, w1)
            xw2 = utils.flat_pos_width(x2, w2)
            seed_type_1 = tf.one_hot(type1, 6)
            seed_xw_1 = tf.one_hot(xw1, 78)
            seed_type_2 = tf.one_hot(type2, 6)
            seed_xw_2 = tf.one_hot(xw2, 78)
            note_type_1.append(seed_type_1)
            note_xw_1.append(seed_xw_1)
            note_type_2.append(seed_type_2)
            note_xw_2.append(seed_xw_2)
            i+=1
        print()
        return (list(map(lambda x: tf.argmax(x, axis=0).numpy(), note_type_1)),
        list(map(lambda x: tf.argmax(x, axis=0).numpy(), note_xw_1)),
        list(map(lambda x: tf.argmax(x, axis=0).numpy(), note_type_2)),
        list(map(lambda x: tf.argmax(x, axis=0).numpy(), note_xw_2)))
