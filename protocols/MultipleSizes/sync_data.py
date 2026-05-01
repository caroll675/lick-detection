import scipy.io as sio
import numpy as np
import pandas as pd

def sync_bpod_video(gpio_file_path, bpod_data_path):
    """
    Returns:
        dataframe mapping video frames into Bpod time.
    """
    parse_gpio_data = pd.read_csv(gpio_file_path, header=None)
    parse_gpio_data.columns = ['pulse1', 'pulse2', 'timestamp']

    sync_pulse_start_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() > 0)].index
    sync_pulse_end_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() < 0)].index - 1
    print(len(sync_pulse_start_frame))

    parse_gpio_data['syncpulse'] = np.where(parse_gpio_data['pulse1'] > 0, 'syncpulse', 'None')

    bpod_data = sio.loadmat(bpod_data_path, squeeze_me=True, struct_as_record=False)
    SessionData = bpod_data['SessionData']


    TrialTypes = SessionData.RewardAmounts
    RewardValveTimes = [0.0544, 0.1952]
    print(TrialTypes)
    TrialStartTimestamp = SessionData.TrialStartTimestamp # trig in matlab scripts 
    TrialEndTimestamp = SessionData.TrialEndTimestamp

    fps = 30
    n_frames = len(parse_gpio_data)

    state = np.full(n_frames, 'None', dtype=object)
    trial_type_col = np.full(n_frames, 'None', dtype=object)
    trial_start_time_col = np.full(n_frames, np.nan, dtype=float)
    bpod_time_s_col = np.full(n_frames, np.nan, dtype=float)
    bpod_time_ms_col = np.full(n_frames, np.nan, dtype=float)


    def to_frames_round(sec, round_mode):
        if round_mode == 'down':
            return int(np.floor(sec * fps))
        elif round_mode == 'up':
            return int(np.ceil(sec * fps))

    def paint(start, seconds, label, round_mode='down'):
        s = int(start)
        e = min(s + to_frames_round(seconds, round_mode), len(state))
        if e > s:
            state[s:e] = label
        return e

    for i, start_frame in enumerate(sync_pulse_start_frame):
        tt = TrialTypes[i]
        s = int(start_frame)

        trial_start_frame = s
        trial_duration_sec = TrialEndTimestamp[i] - TrialStartTimestamp[i]
        trial_end_frame = min(trial_start_frame + int(round(trial_duration_sec * fps)), n_frames)

        trial_type_col[trial_start_frame:trial_end_frame] = tt
        trial_start_time_col[trial_start_frame:trial_end_frame] = TrialStartTimestamp[i]

        frame_idx = np.arange(trial_start_frame, trial_end_frame)
        frame_offset_s = (frame_idx - trial_start_frame) / float(fps)
        # mapping video frames into Bpod time
        bpod_time_s = TrialStartTimestamp[i] + frame_offset_s 

        bpod_time_s_col[trial_start_frame:trial_end_frame] = bpod_time_s
        bpod_time_ms_col[trial_start_frame:trial_end_frame] = bpod_time_s * 1000.0

        if tt == 2:
            s = paint(s, RewardValveTimes[0], 'Reward', 'up')
            s = paint(s, trial_duration_sec - RewardValveTimes[0], 'ITI', 'down')
        if tt == 8:
            s = paint(s, RewardValveTimes[1], 'Reward', 'up')
            s = paint(s, trial_duration_sec - RewardValveTimes[1], 'ITI', 'down')
    parse_gpio_data['state'] = state
    parse_gpio_data['trialtype'] = trial_type_col
    parse_gpio_data['trialstarttime'] = trial_start_time_col
    parse_gpio_data['bpod_time_ms'] = bpod_time_ms_col

    parse_gpio_data.to_csv('tmp.csv', index=False)
    return parse_gpio_data