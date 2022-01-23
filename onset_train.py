import tensorflow as tf
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

def convert_to_data(x):
  frq, difficulty, steps = x['frq'], x['difficulty'], x['steps']
  splited = tf.strings.split(steps, sep=',')
  chart_step_frames = tf.strings.to_number(splited, tf.int32)
  frame_count = tf.shape(frq)[0]
  diffi = tf.repeat(tf.expand_dims(tf.one_hot(difficulty, 5),axis=0), 0 if frame_count is None else frame_count, axis=0)
  chosen_indices = tf.where(chart_step_frames < frame_count)
  chosen_val = tf.gather_nd(chart_step_frames, chosen_indices)
  chart_y = tf.scatter_nd(tf.reshape(chosen_val, [-1,1]), tf.ones(tf.shape(chosen_val)[0]), tf.reshape(frame_count,[1]))
  chart_y = tf.squeeze(tfio.experimental.filter.gaussian(tf.reshape(chart_y * 2 * math.pi,[1,1,-1,1]),7,1))
  return {'frq':frq, 'difficulty':diffi}, chart_y

def extend_frames(x, chart_y):
  lmel_ts, diffi = x['frq'], x['difficulty']
  padded = tf.concat([tf.zeros((7,80,3)), lmel_ts,tf.zeros((7,80,3))],axis=0)
  frame_frq = tf.signal.frame(padded, 15, 1, axis=0)
  return tf.data.Dataset.from_tensor_slices(({'frq':frame_frq, 'difficulty':diffi}, chart_y))

model = models.OnsetModel()
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
      event_frames = sorted(set(map(int,raw_chart.get_frame_list())))
      event_frames = ','.join([str(f) for f in event_frames])
      chart_steps.append(event_frames)
    songs.append(tf.RaggedTensor.from_row_lengths([audio_file, str(music["fillerSec"]), *chart_steps],[1,1,5]))

ds = tf.data.Dataset.from_tensor_slices(tf.stack(songs))
train_size = int(0.8 * len(ds))
test_size = int(0.1 * len(ds))
full_ds = ds.flat_map(read_audio).shuffle(1000, reshuffle_each_iteration=False)
train_ds = full_ds.take(train_size)
test_ds = full_ds.skip(train_size)
val_ds = test_ds.skip(test_size)
test_ds = test_ds.take(test_size)
train_cds = train_ds.map(convert_to_data).cache('train_cache').flat_map(extend_frames).shuffle(1000).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
# test_cds = test_ds.map(convert_to_data).cache('test_cache').flat_map(extend_frames).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
# val_cds = val_ds.map(convert_to_data).cache('val_cache').flat_map(extend_frames).batch(64).prefetch(tf.data.experimental.AUTOTUNE)
model.train(train_cds, epochs=10)
model.save('onset_model_data')