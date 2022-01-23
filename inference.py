import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onset_detector_data', default='onset_model_data', help='the SavedModel directory of the Onset Detector model')
    parser.add_argument('--action_placer_data', default='action_model_data', help='the SavedModel directory of the Action Placer model')
    parser.add_argument('--difficulty', '-d', type=int, choices=[0,1,2,3,4], default=2, help='the difficulty of the generated chart')
    parser.add_argument('input_audio', help='input music file')
    parser.add_argument('output_path', help='output file path. Extension of the file is always made .fbscore')

    args = parser.parse_args()

    import numpy as np
    import models
    import utils
    import os
    from utils import get_mel_spectrogram
    import frame_based_score

    onset_model = models.OnsetModel(load_file_path=args.onset_detector_data)
    action_model = models.NoteActionModel(load_file_path=args.action_placer_data)

    ifp = args.input_audio
    ofp = os.path.splitext(args.output_path)[0] + '.fbscore'
    diffi = args.difficulty

    print('Retrieving mel spectrogram from the audio sample...')
    audio_sample, fs = utils.get_audio_sample(ifp)
    frame_step = fs // 100
    audio_lmel80 = np.stack([get_mel_spectrogram(audio_sample, 1024, frame_step, fs),
            get_mel_spectrogram(audio_sample, 2048, frame_step, fs),
            get_mel_spectrogram(audio_sample, 4096, frame_step, fs)], axis=2)

    print('Finding onset frames...')
    steps = onset_model.query(audio_lmel80=audio_lmel80, difficulty=diffi)
    print('Generating note actions...')
    actions = action_model.query(audio_lmel80=audio_lmel80, difficulty=diffi, steps=steps)
    print('Saving file...')
    frame_based_score.save_score(ofp, steps, actions)