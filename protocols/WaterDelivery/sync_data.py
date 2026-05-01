import scipy.io as sio
import numpy as np
import pandas as pd

def sync_bpod_video(gpio_file_path, bpod_data_path):
    """
    Returns:
        csv file: synced data
    """
    parse_gpio_data = pd.read_csv(gpio_file_path, header=None)
    parse_gpio_data.columns = ['pulse1', 'pulse2', 'timestamp']
    # load the sync pulse start and end in 24 hour clock system
    sync_pulse_start_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() > 0)].index
    sync_pulse_end_frame = parse_gpio_data[(parse_gpio_data['pulse1'].diff() < 0)].index-1
    print('total number of trials: ' + str(len(sync_pulse_start_frame)))
    sync_pulse_start_timestamp = parse_gpio_data.loc[sync_pulse_start_frame, 'timestamp']
    sync_pulse_end_timestamp =  parse_gpio_data.loc[sync_pulse_end_frame, 'timestamp']
    parse_gpio_data['syncpulse'] = np.where(parse_gpio_data['pulse1'] > 0, 'syncpulse', 'None')

    # Load the .mat file. Use the optional 'struct_as_record=False' to load structs as objects,
    # which can sometimes simplify access.
    # squeeze_me=True can help flatten single-element arrays automatically.
    bpod_data = sio.loadmat(bpod_data_path, squeeze_me=True, struct_as_record=False)

    # Access the top-level 'SessionData' struct.
    # It often loads as a single-element object array.
    SessionData = bpod_data['SessionData'] 
    field_names = [attr for attr in dir(SessionData) if not attr.startswith('__')]
    # print(f"SessionData contains the following fields: {field_names}")
    # Access nested fields. The exact access depends on the structure.
    # With squeeze_me=True, you often don't need [0, 0].

    RewardValveTime = 0.0883 # reward valve time
    TrialStartTimestamp = SessionData.TrialStartTimestamp # in sec
    TrialEndTimestamp = SessionData.TrialEndTimestamp
    fps = 30

    # Initialize arrays for state, trial type, and trial start time
    # Since there's only one trial type, we'll use a constant value (0) for all trials
    state = np.full(len(parse_gpio_data), 'None', dtype=object)
    trial_type_col = np.full(len(parse_gpio_data), 'None', dtype=object)
    trial_start_time_col = np.full(len(parse_gpio_data), 'None', dtype=object)

    def to_frames_round(sec, round):
        if round == 'down':
            return int(np.floor(sec * fps))
        elif round == 'up':
            return int(np.ceil(sec * fps))

    def paint(start, seconds, label, round='down'):
        s = int(start)
        e = min(s + to_frames_round(seconds, round), len(state))
        if e > s:
            state[s:e] = label
        return e  # next start

    # bpod might take some time to transition to the next trial so there are some empty frames between trials?
    # Since there's only one trial type, we use a constant value (0) for all trials
    for i, start_frame in enumerate(sync_pulse_start_frame):
        s = int(start_frame)

        trial_start_frame = s # syncpulse start frame is also the trial start frame
        trial_duration_sec = TrialEndTimestamp[i] - TrialStartTimestamp[i]
        trial_end_frame = trial_start_frame + int(round(trial_duration_sec*fps)) 
        # All trials are the same type (0 = free reward)
        trial_type_col[trial_start_frame:trial_end_frame] = 0
        trial_start_time_col[trial_start_frame:trial_end_frame] = TrialStartTimestamp[i]

        # Single trial type: free reward
        s = paint(s,  RewardValveTime, 'Reward', 'up')  
        s = paint(s, trial_duration_sec - RewardValveTime, 'ITI', 'down')


    parse_gpio_data['state'] = state
    parse_gpio_data['trialtype'] = trial_type_col
    parse_gpio_data['trialstarttime'] = trial_start_time_col
    parse_gpio_data.to_csv('tmp.csv')
    return parse_gpio_data

# gpio_file_path = r"\\140.247.90.110\homes2\Carol\VideoData\CC4_20251007_114439_gpio1.csv"
# bpod_data_path = r"\\140.247.90.110\homes2\Carol\FakeSubject_CombinedStimOdorTask_20251008_175512.mat"

# sync_bpod_video(gpio_file_path, bpod_data_path)

# modify # [change here]