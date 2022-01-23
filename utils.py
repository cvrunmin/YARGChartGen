import tensorflow as tf
import tensorflow_io as tfio


def get_audio_sample(fp):
    audio = tfio.audio.AudioIOTensor(fp)
    audio_sample = tf.cast(tf.math.reduce_mean(audio.to_tensor(), axis=1), tf.float32)
    fs = audio.rate.numpy()
    return (audio_sample, fs)


def get_mel_spectrogram(audio_sample, frame_length, frame_step=441, fs=44100):
    f_min, f_max, num_mel_bins = 27.5, 16000.0, 80
    audio_frq = tf.signal.stft(audio_sample, frame_length=frame_length, frame_step=frame_step, pad_end=True)
    mel_transform = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, audio_frq.shape[-1], fs, f_min,
      f_max)
    return tf.math.log(tf.tensordot(tf.abs(audio_frq), mel_transform, 1)+1e-16)


def flat_pos_width(x, width):
    tbl = [0,12,23,33,42,50,57,63,68, 72, 75, 77]
    return tbl[x] + width - 1

def get_pos_width_from_flatten(fl):
    from bisect import bisect
    tbl = [0,12,23,33,42,50,57,63,68, 72, 75, 77]
    idx = bisect(tbl, fl)
    if idx == 0:
        return 0, 1
    return idx - 1, fl - tbl[idx - 1] + 1