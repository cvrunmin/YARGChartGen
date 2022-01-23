import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help='Set the train process to be CPU only')
args = parser.parse_args()

if args.cpu:
  print('try disabling gpu...')
  tf.config.set_visible_devices([], 'GPU')

import tensorflow_io as tfio
import models
from utils import get_mel_spectrogram
import math, json, sus_score


def filter_some_notes(note_event):
  if note_event.type == 1 and note_event.subtype > 2:
    return False
  if note_event.x_pos < 2 or note_event.x_pos > 13:
    return False
  if note_event.type == 3 and note_event.subtype == 5:
    return False
  if note_event.type == 5 and note_event.subtype in [2,5,6]:
    return False
  return True

def flat_pos_width(x, width):
    tbl = tf.constant([0,12,23,33,42,50,57,63,68, 72, 75, 77])
    return tbl[x] + width - 1

def get_pos_width_from_flatten(fl):
    from bisect import bisect
    tbl = [0,12,23,33,42,50,57,63,68, 72, 75, 77]
    idx = bisect(tbl, fl)
    if idx == 0:
        return 0, 1
    return idx - 1, fl - tbl[idx - 1] + 1

def get_detailed_frame_info(raw_sus: sus_score.RawSusChart):
    modifier_map = dict()
    tick_per_beat = 480
    beat = raw_sus.beat_counts[0]
    speed = raw_sus.bpms[0]
    for item in raw_sus.modifier_note_list:
        key = f"M{item.measure}T{item.offset_tick}X{item.x_pos-2}"
        modifier_map[key] = item
    frames = dict()
    for item in raw_sus.short_note_list:
        tick = item.measure * beat.count * tick_per_beat + item.offset_tick
        time = int(tick * 6000 / tick_per_beat / speed.bpm)
        key = f"M{item.measure}T{item.offset_tick}X{item.x_pos-2}"
        if key in modifier_map:
            frames.setdefault(time, list()).append(f'{item.x_pos-2}%{item.width}%2%{0 if item.x_pos < 8 else 1}')
        else:
            frames.setdefault(time, list()).append(f'{item.x_pos-2}%{item.width}%1%{0 if item.x_pos < 8 else 1}')
    for item in raw_sus.long_note_list:
        tick = item.measure * beat.count * tick_per_beat + item.offset_tick
        time = int(tick * 6000 / tick_per_beat / speed.bpm)
        if item.subtype == 1:
            subtype = 3
        elif item.subtype == 2:
            subtype = 5
        elif item.subtype == 3:
            subtype = 4
        else:
            continue
        frames.setdefault(time, list()).insert(0,f'{item.x_pos-2}%{item.width}%{subtype}%{item.channel}')
    frames = sorted(frames.items(), key=lambda x:x[0])
    return '|'.join([str(k)+';'+','.join(v) for k,v in frames])

def read_audio(x):
  audio_file, offset_sec, steps = x[0], float(x[1]), x[2]
  audio = tfio.audio.AudioIOTensor(audio_file, dtype=tf.int16)
  audio_sample = tf.realdiv(tf.cast(tf.math.reduce_mean(audio.to_tensor(), axis=1), tf.float32), tf.constant(32768.0))
  offset = tf.cast(offset_sec * tf.cast(audio.rate, tf.float32), tf.int32)
  audio_sample = tf.slice(audio_sample, tf.reshape(offset, [1]), tf.shape(audio_sample) - offset)
  lmel_ts = tf.stack([get_mel_spectrogram(audio_sample, 1024),
                      get_mel_spectrogram(audio_sample, 2048),
                      get_mel_spectrogram(audio_sample, 4096)], axis=2)
  del audio, audio_sample
  frq = tf.repeat(tf.expand_dims(lmel_ts, 0), 5,axis=0)
  diffi = list(range(0,5))
  return tf.data.Dataset.from_tensor_slices({'frq':frq, 'difficulty':diffi, 'steps':steps})

def frame_string_to_usable_data(steps):
    with tf.device('/CPU:0'):
        splitted_steps = tf.strings.split(steps, sep='|')
        splitted_kv = tf.strings.split(splitted_steps, sep=';')
        frames_no = tf.strings.to_number(tf.squeeze(splitted_kv[:,:1],axis=1), out_type=tf.int32)
        action_by_frame = tf.strings.split(tf.squeeze(splitted_kv[:,1:], axis=1),sep=',')
        return tf.data.Dataset.from_tensor_slices({'frame':frames_no, 'data': tf.strings.to_number(tf.strings.split(action_by_frame, '%'), out_type=tf.int32).to_tensor()})

def convert_to_data(x):
  frq, difficulty, steps = x['frq'], x['difficulty'], x['steps']
  padded = tf.concat([tf.zeros((7,80,3)), frq,tf.zeros((7,80,3))],axis=0)
  with tf.device('/CPU:0'):
    frame_detail_dataset = frame_string_to_usable_data(steps)
    frame_frq, c0_type, c0_xw, c1_type, c1_xw = tf.TensorArray(tf.float32, size=0, dynamic_size=True), tf.TensorArray(tf.float32, size=0, dynamic_size=True), tf.TensorArray(tf.float32, size=0, dynamic_size=True), tf.TensorArray(tf.float32, size=0, dynamic_size=True), tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    frame_idcs = tf.TensorArray(tf.int32,size=0,dynamic_size=True)
    c0_long_press, c1_long_press = False, False
    array_idx = 0
    c0_type=c0_type.write(array_idx, tf.one_hot(0,6))
    c0_xw=c0_xw.write(array_idx, tf.one_hot(0,78))
    c1_type=c1_type.write(array_idx, tf.one_hot(0,6))
    c1_xw=c1_xw.write(array_idx, tf.one_hot(0,78))
    frame_idcs=frame_idcs.write(array_idx, -(1<<30))
    array_idx+=1
    for entry in frame_detail_dataset:
      frame_idx, data = entry['frame'], entry['data']
      c0_assign, c1_assign = False, False
      # c0_assign.assign(False)
      # c1_assign.assign(False)
      if frame_idx >= tf.shape(frq)[0]:
        frame_frq=frame_frq.write(array_idx-1,tf.zeros([15,80,3]))
      else:
        frame_frq=frame_frq.write(array_idx-1, tf.reshape(padded[frame_idx:frame_idx+15], (15,80,3)))
      # frame_idcs.write(array_idx, frame_idx)
      # frame_frq.append(padded[frame_idx:frame_idx+15])
      frame_idcs=frame_idcs.write(array_idx,frame_idx)
      for datum in data:
        xw = tf.one_hot(flat_pos_width(datum[0], datum[1]), 78)
        type_idx = datum[2]
        type = tf.one_hot(type_idx, 6)
        if not c0_assign and datum[3] == 0:
          c0_type=c0_type.write(array_idx, type)
          c0_xw=c0_xw.write(array_idx, xw)
          if type_idx == 3:
            if c0_long_press:
              tf.print(f'too many slide start at {frame_idx} C0')
            c0_long_press = (True)
          elif type_idx == 5:
            c0_long_press = (False)
          c0_assign=(True)
        elif not c1_assign and datum[3] == 0 and type_idx < 3:
          c1_type=c1_type.write(array_idx, type)
          c1_xw=c1_xw.write(array_idx, xw)
          c1_assign = (True)
        elif not c1_assign and datum[3] == 1:
          c1_type=c1_type.write(array_idx, type)
          c1_xw=c1_xw.write(array_idx, xw)
          if type_idx == 3:
            if c1_long_press:
              tf.print(f'too many slide start at {frame_idx} C1')
            c1_long_press = (True)
          elif type_idx == 5:
            c1_long_press=(False)
          c1_assign=(True)
        elif not c0_assign and datum[3] == 1 and type_idx < 3:
          c0_type=c0_type.write(array_idx, type)
          c0_xw=c0_xw.write(array_idx, xw)
          c0_assign=(True)
      if not c0_assign:
        c0_type=c0_type.write(array_idx, tf.one_hot(0, 6))
        c0_xw=c0_xw.write(array_idx, tf.one_hot(0,78))
      if not c1_assign:
        c1_type=c1_type.write(array_idx, tf.one_hot(0, 6))
        c1_xw=c1_xw.write(array_idx, tf.one_hot(0,78))
      array_idx += 1
    frame_idcs=frame_idcs.write(array_idx,1<<30)
    frame_ts = frame_idcs.stack()
    delta_frame = tf.subtract(frame_ts[1:], frame_ts[:-1])
    backward_delta_frame = tf.cast(delta_frame[:-1],tf.float32)
    forward_delta_frame = tf.cast(delta_frame[1:],tf.float32)
    backward_delta_frame = tf.multiply(backward_delta_frame, tf.constant(0.2))
    forward_delta_frame = tf.multiply(forward_delta_frame, tf.constant(0.2))
    backward_delta_frame = tf.scalar_mul(1 / math.log(2), tf.math.log(backward_delta_frame))
    forward_delta_frame = tf.scalar_mul(1 / math.log(2), tf.math.log(forward_delta_frame))
    backward_delta_frame = tf.clip_by_value(tf.cast(tf.math.floor(backward_delta_frame),tf.int32), 0, 7)
    forward_delta_frame = tf.clip_by_value(tf.cast(tf.math.floor(forward_delta_frame),tf.int32), 0, 7)
    backward_delta_one_hot = tf.one_hot(backward_delta_frame, 8)
    forward_delta_one_hot = tf.one_hot(forward_delta_frame, 8)
    nti1 = c0_type.stack()[:-1]
    nti2 = c1_type.stack()[:-1]
    nxw1 = c0_xw.stack()[:-1]
    nxw2 = c1_xw.stack()[:-1]
    # time_interval = tf.stack([backward_delta_one_hot, forward_delta_one_hot], axis=1)
    frq_ls = frame_frq.stack()
    frame_count = tf.shape(backward_delta_one_hot)[0]
    diffi = tf.repeat(tf.expand_dims(tf.one_hot(difficulty, 5),axis=0), 0 if frame_count is None else frame_count, axis=0)
    return tf.data.Dataset.from_tensor_slices(({'frq':frq_ls,
        'difficulty':diffi,
        'note_type_1': nti1,
        'note_xw_1': nxw1,
        'note_type_2': nti2,
        'note_xw_2': nxw2,
        'time_interval_before': backward_delta_one_hot,
        'time_interval_after': forward_delta_one_hot
    }, \
    {
            'note_type_out_1': c0_type.stack()[1:],
            'note_xw_out_1': c0_xw.stack()[1:],
            'note_type_out_2': c1_type.stack()[1:],
            'note_xw_out_2': c1_xw.stack()[1:]
    }))


model = models.NoteActionModel(lstm_cell=512)
songs = []
with open("./prsk_dataset/musics.json", 'r', encoding='utf-8') as f:
  musics_json = json.load(f)
  difficulty_label = ['easy','normal','hard','expert','master']
  for music in musics_json["data"]:
    audio_file = f'./prsk_dataset/{music["id"]:04d}_01/{music["id"]:04d}_01.flac'
    chart_steps = []
    for i in range(0,5):
      path = f'./prsk_dataset/{music["id"]:04d}_01/{difficulty_label[i]}.sus'
      try:
        raw_chart = sus_score.read_sus(path, encoding='utf-8', filter=filter_some_notes)
      except UnicodeDecodeError:
        raw_chart = sus_score.read_sus(path, encoding='shift-jis', filter=filter_some_notes)
      chart_steps.append(get_detailed_frame_info(raw_chart))
    songs.append(tf.RaggedTensor.from_row_lengths([audio_file, str(music["fillerSec"]), *chart_steps],[1,1,5]))

ds = tf.data.Dataset.from_tensor_slices(tf.stack(songs))
train_size = int(0.8 * len(ds))
test_size = int(0.1 * len(ds))
full_ds = ds.flat_map(read_audio).shuffle(1000, reshuffle_each_iteration=False)
train_ds = full_ds.take(train_size)
test_ds = full_ds.skip(train_size)
val_ds = test_ds.skip(test_size)
test_ds = test_ds.take(test_size)
train_cds = train_ds.flat_map(convert_to_data).cache('step_train_cache').batch(64).prefetch(tf.data.experimental.AUTOTUNE)
# test_cds = test_ds.flat_map(convert_to_data).cache('step_test_cache').batch(64).prefetch(tf.data.experimental.AUTOTUNE)
# val_cds = val_ds.flat_map(convert_to_data).cache('step_val_cache').batch(64).prefetch(tf.data.experimental.AUTOTUNE)
model.train(train_cds, epochs=10)
model.save('action_model_data')