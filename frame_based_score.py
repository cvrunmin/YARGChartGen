from utils import get_pos_width_from_flatten as break_xw


def save_score(file_path: str, frames: list, steps_action: tuple):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write('#FRAME BASED SCORE\n')
        file.write('#VERSION 1.0\n')
        for frame_action in zip(frames,*steps_action):
            frame = frame_action[0]
            t1 = frame_action[1]
            xw1 = frame_action[2]
            x1, w1 = break_xw(xw1)
            t2 = frame_action[3]
            xw2 = frame_action[4]
            x2, w2 = break_xw(xw2)
            file.write(f'{frame}:T{t1}X{x1}W{w1},T{t2}X{x2}W{w2}\n')