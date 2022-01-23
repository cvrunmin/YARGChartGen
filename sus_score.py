# take a look at sus-analyzer?
import os, string, re, dataclasses, typing
from typing import Callable, Dict, List

tick_per_beat = 480

bpm_def_pattern = re.compile('BPM(\\d{2}):\\s*(\\d+(\\.\\d+)?)')
bpm_apply_pattern = re.compile('(\\d{3})08:\\s*(\\d{2})')
beat_unit_pattern = re.compile('(\\d{3})02:\\s*(\\d+)')
note_event_pattern = re.compile('(?P<measure>\\d{3})(?P<type>[1-5])(?P<left>[0-9a-zA-Z])(?P<channel>[0-9a-zA-Z])?:(\\s*[0-9a-zA-Z][0-9a-zA-Z])+\\s*')

@dataclasses.dataclass
class RawNoteEventInfo:
    measure: int
    offset_tick: int
    x_pos: int
    width: int
    type: int
    subtype: int
    channel: int = 0


@dataclasses.dataclass
class BeatCountInfo:
    measure: int
    count: int


@dataclasses.dataclass
class BpmInfo:
    measure: int
    bpm: float


class RawSusChart:
    beat_counts : List[BeatCountInfo]
    bpms : List[BpmInfo]
    note_event_key : List[str]
    short_note_list: List[RawNoteEventInfo]
    long_note_list: List[RawNoteEventInfo]
    modifier_note_list: List[RawNoteEventInfo]

    def __init__(self, beat_counts: list, bpm_def: dict, bpm_appling: list, note_events: List[RawNoteEventInfo], filter: Callable[[RawNoteEventInfo], bool] = lambda x:True):
        self.beat_counts = beat_counts
        self.bpms = list()
        self.note_event_key = list()
        self.short_note_list = list()
        self.long_note_list = list()
        self.modifier_note_list = list()
        for item in bpm_appling:
            if 'bpm_id' in item and 'measure' in item:
                if item['bpm_id'] in bpm_def:
                    self.bpms.append(BpmInfo(item['measure'], bpm_def[item["bpm_id"]]))
        for item in note_events:
            if item.type in range(1, 6):
                if not filter(item): continue
                if item.type == 1:
                    self.short_note_list.append(item)
                elif item.type == 5:
                    self.modifier_note_list.append(item)
                else:
                    self.long_note_list.append(item)
                self.note_event_key.append(f"M{item.measure}T{item.offset_tick}")
        self.note_event_key = list(set(self.note_event_key))

    
    def get_frame_list(self):
        frames = list()
        changing_beats = True
        changing_speed = True
        beat = self.beat_counts[0]
        speed = self.bpms[0]
        if len(self.beat_counts) == 1:
            changing_beats = False
        if len(self.bpms) == 1:
            changing_speed = False
        for event in self.note_event_key:
            measure = int(event[1:].split('T')[0])
            offset_tick = int(event.split('T')[1])
            tick = measure * beat.count * tick_per_beat + offset_tick
            time = tick * 6000 / tick_per_beat / speed.bpm
            frames.append(time)
        # TODO: handle changing beat per measure, changing bpm
        return frames
                

# note:
# note type 5 usually act as modifier event
# 1,3,4: flick (striaght, left, right)
# 2: ease in slide
# 5,6: ease out slide
# note type 3 defines a slides action
# note type 1 with subtype 3 tells slide waypoint to locate at center of sliding rail
# note type 1 with subtype 4 is skill action
# lane number starts with 2 instead of 0. ends with d
# sometimes slide start note may also appear in short note, we should be able to merge them

def read_sus(file_path, encoding='utf-8', filter: Callable[[RawNoteEventInfo], bool] = lambda x:True):
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()

        beat_def_list = list()
        bpm_def_dict = dict()
        bpm_apply_list = list()
        note_event_list = list()
        for line in lines:
            if line is None or not line:
                continue
            if not line.startswith('#'):
                continue
            line_content = line[1:]
            if "TITLE" in line_content \
                or "SUBTITLE" in line_content \
                or "ARTIST" in line_content \
                or "GENRE" in line_content \
                or "DESIGNER" in line_content \
                or "DIFFICULTY" in line_content \
                or "PLAYLEVEL" in line_content \
                or "SONGID" in line_content \
                or "WAVE" in line_content \
                or "WAVEOFFSET" in line_content \
                or "JACKET" in line_content \
                or "BACKGROUND" in line_content \
                or "MOVIE" in line_content \
                or "MOVIEOFFSET" in line_content \
                or "BASEBPM" in line_content \
                or "REQUEST" in line_content \
                or "HISPEED" in line_content \
                or "MEASUREHS" in line_content \
                or "TIL" in line_content:
                continue # skip all metadata
            result = beat_unit_pattern.match(line_content)
            if result:
                beat_pos = int(result.group(1))
                beat_count = int(result.group(2))
                beat_def_list.append(BeatCountInfo(beat_pos, beat_count))
                continue
            result = bpm_def_pattern.match(line_content)
            if result:
                bpm_id = int(result.group(1))
                bpm_value = float(result.group(2))
                bpm_def_dict[bpm_id] = bpm_value
                continue
            result = bpm_apply_pattern.match(line_content)
            if result:
                bpm_pos = int(result.group(1))
                bpm_id = int(result.group(2))
                bpm_apply_list.append({'measure': bpm_pos, 'bpm_id': bpm_id})
                continue
            result = note_event_pattern.match(line_content)
            if result:
                measure = int(result.group("measure"))
                type = int(result.group("type"))
                left_pos = int(result.group("left"), base=36)
                c = result.group("channel")
                channel = 0 if c is None else int(c, base=36)
                data_str = line_content.split(':')[1].strip()
                if len(data_str) % 2 != 0:
                    print(f'wrong length of data part ({len(data_str)}): \"{line_content}\"')
                    continue
                subdivided_count = len(data_str) // 2
                note_step = 4 / subdivided_count
                for i in range(0, subdivided_count):
                    note_str = data_str[i*2:(i+1)*2]
                    if note_str[0] == '0':
                        continue # skip no event position
                    note_subtype = int(note_str[0], base=36)
                    note_width = int(note_str[1], 36)
                    note_event_list.append(RawNoteEventInfo(measure, int(tick_per_beat * note_step * i), left_pos, note_width, type, note_subtype, channel))
                continue
            print(f'unknown line content: {line_content}')
        return RawSusChart(beat_def_list, bpm_def_dict, bpm_apply_list, note_event_list, filter)
            