# === lickdetection_full.py ===
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle
import sync_data  # your existing module


# ----------------------------
# ROI (Region of Interest) selection from a representative frame (default: middle)
# ----------------------------
def select_roi(
    video_path,
    *,
    source_frame="middle",         # "middle" | "first" | "last" (ignored if frame_index is set)
    frame_index=None,              # exact frame to use for ROI selection
    skip_head_seconds=0.0,         # ignore head when computing "middle"
    skip_tail_seconds=0.0,         # ignore tail when computing "middle"
    from_center=False,
    show_crosshair=True,
    window_title=None,
):
    """
    Open a single frame for user ROI selection.

    Returns:
        (roi_tuple, frame_index) or (None, None)
        where roi_tuple = (x, y, w, h)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    if total_frames <= 0:
        print("Error: Video has zero frames.")
        cap.release()
        return None, None

    # Compute target frame
    if frame_index is not None:
        target = max(0, min(total_frames - 1, int(frame_index)))
    else:
        if source_frame == "first":
            target = 0
        elif source_frame == "last":
            target = total_frames - 1
        else:
            # "middle" inside the usable window (skip head/tail if requested)
            if fps > 0:
                start_det = int(max(0, round(skip_head_seconds * fps)))
                end_det = int(max(0, total_frames - 1 - round(skip_tail_seconds * fps)))
                if end_det <= start_det:
                    start_det, end_det = 0, total_frames - 1
            else:
                start_det, end_det = 0, total_frames - 1
            target = (start_det + end_det) // 2

    # Seek and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    if not ok:
        # Fallback to first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok:
            print("Error: Could not read a frame for ROI selection.")
            cap.release()
            return None, None

    cap.release()

    title = window_title or f"Select ROI (frame {target+1}/{total_frames})"
    roi = cv2.selectROI(title, frame, fromCenter=from_center, showCrosshair=show_crosshair)
    cv2.destroyWindow(title)

    x, y, w, h = roi
    if w > 0 and h > 0:
        return (int(x), int(y), int(w), int(h)), int(target)
    return None, None


# ----------------------------
# Lick detection with optional GPU (Graphics Processing Unit) acceleration and video saving
# ----------------------------
def detect_licks(
    video_path,
    roi,
    threshold= 25,
    min_movement_percent=25,
    cooldown_frames=2,
    show_video_with_licks=False,
    playback_speed=0.5,
    synced_gpio_file=None,
    save_video_path=None,
    save_codec=None,
    save_fps=None,
    start_at_frame=None,           # where on-screen playback should start (e.g., middle frame)
    save_full_video=True,          # if True, still save annotated video from frame 0 → end
    skip_head_seconds=0.0,         # detection disabled before this time
    skip_tail_seconds=0.0,         # detection disabled in the last N seconds
):
    """
    Detect mouse licks using frame differencing within an ROI. Uses PyTorch with CUDA on the GPU
    if available to accelerate grayscale, blur, absdiff, threshold, dilation, and counting.

    Returns:
        (lick_timestamps, lick_frames_array, video_duration_seconds, fps)
    """
    if synced_gpio_file is None:
        print("Error: synced_gpio_file is required.")
        return None, None, 0, 0

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        print("Error: ROI has non-positive width/height.")
        return None, None, 0, 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None, None, 0, 0

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wait_key_delay = 1 if fps == 0 else max(1, int(1000 / (fps * playback_speed)))

    # Detection window (skip head/tail for detection only)
    if fps > 0:
        det_start = int(max(0, round(skip_head_seconds * fps)))
        det_end = int(max(0, total_frames - 1 - round(skip_tail_seconds * fps)))
        if det_end <= det_start:
            det_start, det_end = 0, max(0, total_frames - 1)
    else:
        det_start, det_end = 0, max(0, total_frames - 1)

    # Decide display vs processing start
    display_start_frame = int(start_at_frame) if start_at_frame is not None else 0
    display_start_frame = max(0, min(total_frames - 1, display_start_frame))
    processing_start = 0 if save_full_video else display_start_frame

    # Seek to processing_start before reading first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, processing_start)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame at the requested start.")
        cap.release()
        return None, None, 0, 0

    # Threshold in pixels from percentage
    roi_area = w * h
    min_changed_pixels = roi_area * (min_movement_percent / 100.0)

    # Video writer (edited output)
    writer = None
    if save_video_path is not None:
        save_path = Path(save_video_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_codec is None:
            ext = save_path.suffix.lower()
            if ext in (".mp4", ".m4v"):
                fourcc_str = "mp4v"
            elif ext in (".avi",):
                fourcc_str = "XVID"
            else:
                fourcc_str = "mp4v"
        else:
            fourcc_str = save_codec
        out_fps = float(save_fps) if save_fps is not None else (fps if fps > 0 else 30.0)
        writer = cv2.VideoWriter(
            str(save_path),
            cv2.VideoWriter_fourcc(*fourcc_str),
            out_fps,
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            print(f"Warning: Could not open VideoWriter for {save_path}. Video will not be saved.")
            writer = None

    # GPU path via PyTorch if available
    use_torch_cuda = False
    F = None
    try:
        import torch
        import torch.nn.functional as F  # noqa
        use_torch_cuda = torch.cuda.is_available()
    except Exception:
        use_torch_cuda = False

    overlay_needed = show_video_with_licks or (writer is not None)

    if use_torch_cuda:
        device = torch.device("cuda")
        k = 21  # Gaussian kernel size, matches OpenCV (21,21)
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8  # approx OpenCV sigma when sigma=0

        def _gaussian_kernel_2d(ksize, sigma, device):
            ax = torch.arange(ksize, device=device, dtype=torch.float32) - (ksize // 2)
            g1 = torch.exp(-(ax**2) / (2 * sigma**2))
            g1 = g1 / g1.sum()
            g2 = torch.outer(g1, g1)
            return (g2 / g2.sum()).unsqueeze(0).unsqueeze(0)  # [1,1,k,k]

        gauss = _gaussian_kernel_2d(k, sigma, device)

        def _to_gray_tensor(bgr_roi_np):
            # BGR uint8 -> gray float tensor on GPU [1,1,H,W] in [0,1]
            t = torch.from_numpy(bgr_roi_np).to(device=device, dtype=torch.float32)  # H,W,3
            b, g, r = t[..., 0], t[..., 1], t[..., 2]
            gray = 0.114 * b + 0.587 * g + 0.299 * r
            return (gray / 255.0).unsqueeze(0).unsqueeze(0)

        prev_roi = prev_frame[y:y+h, x:x+w]
        prev_t = _to_gray_tensor(prev_roi)
        prev_blur_t = F.conv2d(prev_t, gauss, padding=k // 2)
        print("GPU acceleration: ON (PyTorch CUDA)")
    else:
        prev_roi_gray = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        prev_roi_blur = cv2.GaussianBlur(prev_roi_gray, (21, 21), 0)
        print("GPU acceleration: OFF (CPU OpenCV path)")

    lick_timestamps = []
    lick_frames_array = np.zeros(len(synced_gpio_file), dtype=int)
    in_cooldown = 0
    frame_num = processing_start  # absolute frame index to keep alignment with synced_gpio_file

    # Main loop
    while True:
        # Stop conditions
        if frame_num >= len(synced_gpio_file):
            break
        ret, frame = cap.read()
        if not ret:
            break

        current_time = (frame_num / fps) if fps > 0 else 0.0
        current_frame_idx = frame_num
        frame_num += 1

        vis_frame = frame.copy() if overlay_needed else None
        roi_frame = frame[y:y+h, x:x+w]

        if use_torch_cuda:
            cur_t = _to_gray_tensor(roi_frame)
            cur_blur_t = F.conv2d(cur_t, gauss, padding=k // 2)
            diff_t = (cur_blur_t - prev_blur_t).abs()

            # Threshold in [0,1] scaled to 0..255 like OpenCV
            mask = (diff_t * 255.0) > threshold

            # Dilation (two iterations ~ iterations=2)
            dil1 = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
            dil2 = F.max_pool2d(dil1, kernel_size=3, stride=1, padding=1)

            changed_pixels = int(dil2.sum().item())
            movement_detected = changed_pixels > min_changed_pixels
            prev_blur_t = cur_blur_t

            dilated_np = (dil2.squeeze().detach().to("cpu").numpy() * 255).astype(np.uint8) if overlay_needed else None
        else:
            roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            roi_blur = cv2.GaussianBlur(roi_gray, (21, 21), 0)
            frame_diff = cv2.absdiff(prev_roi_blur, roi_blur)
            _, thresh_img = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
            dilated_np = cv2.dilate(thresh_img, None, iterations=2)
            changed_pixels = cv2.countNonZero(dilated_np)
            movement_detected = changed_pixels > min_changed_pixels
            prev_roi_blur = roi_blur

        # Disable detection outside the desired window
        detection_enabled = (det_start <= current_frame_idx <= det_end)

        # Centroid for green dot (visual only)
        largest_contour = None
        if overlay_needed and dilated_np is not None:
            contours, _ = cv2.findContours(dilated_np.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)

        # Lick detection with cooldown (only if detection_enabled)
        if detection_enabled and movement_detected and in_cooldown == 0:
            lick_timestamps.append(current_time)
            lick_frames_array[current_frame_idx] = 1
            in_cooldown = cooldown_frames

            if overlay_needed and largest_contour is not None and vis_frame is not None:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(vis_frame, (x + cx, y + cy), 3, (0, 255, 0), -1)
        elif in_cooldown > 0:
            in_cooldown -= 1

        # Overlays (ROI box, counters, per-frame metadata)
        if overlay_needed and vis_frame is not None:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(vis_frame, f"Frame: {current_frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis_frame, f"Licks: {len(lick_timestamps)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            try:
                this_state = str(synced_gpio_file['state'].iloc[current_frame_idx])
                this_sync = str(synced_gpio_file['syncpulse'].iloc[current_frame_idx])
                this_trial_type = str(synced_gpio_file['trialtype'].iloc[current_frame_idx])
            except Exception:
                this_state = this_sync = this_trial_type = 'None'

            if this_trial_type != 'None':
                cv2.putText(vis_frame, f"TrialType: {this_trial_type}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if this_sync != 'None':
                cv2.putText(vis_frame, this_sync, (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            if this_state != 'None':
                cv2.putText(vis_frame, this_state, (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Show window only after we reach the requested display start
            if show_video_with_licks and current_frame_idx >= display_start_frame:
                cv2.imshow("Lick Detection Video", vis_frame)
                if cv2.waitKey(wait_key_delay) & 0xFF == ord('q'):
                    break

        # Write edited frame if saving
        if writer is not None:
            writer.write(vis_frame if vis_frame is not None else frame)

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    video_duration_seconds = (total_frames / fps) if fps > 0 else 0.0
    return lick_timestamps, lick_frames_array, video_duration_seconds, fps


# ----------------------------
# Raster plotting (no globals; pass fps, mouse_id, date)
# ----------------------------
def plot_lick_raster(
    synced_gpio_file,
    fps,
    mouse_id,
    date,
    fig_root_dir=Path(r"\\140.247.90.110\homes2\Carol\LickData")
):
    """
    Plots a raster of lick times per trial type. Expects synced_gpio_file to contain:
    'lickframe' (0/1 per frame), 'trialtype', 'trialstarttime'.
    """
    df = synced_gpio_file.copy()
    df = df[df['trialtype'] != 'None']

    if 'lickframe' not in df.columns:
        raise ValueError("synced_gpio_file must have a 'lickframe' column (0/1 per frame).")

    unique_trial_types = sorted(df['trialtype'].unique(), key=lambda v: str(v))

    all_lick_events = {}
    for tt in unique_trial_types:
        tt_df = df[df['trialtype'] == tt].copy()
        trial_starts = tt_df['trialstarttime'].unique()
        all_lick_events[str(tt)] = {}
        for trial_idx, t_start in enumerate(trial_starts):
            single_trial = tt_df[tt_df['trialstarttime'] == t_start]
            lick_frames_in_trial = single_trial.index[single_trial['lickframe'] == 1].to_numpy()
            if lick_frames_in_trial.size > 0:
                trial_start_frame = int(single_trial.index.min())
                rel_times = (lick_frames_in_trial - trial_start_frame) / float(fps)
                all_lick_events[str(tt)][trial_idx] = rel_times.tolist()

    event_positions, event_colors, y_labels = [], [], []
    palette = np.array([
        [41, 114, 112],
        [230, 109, 80],
        [231, 198, 107],
        [138, 176, 124],
        [41, 157, 143],
    ], dtype=float) / 255.0

    for i, tt in enumerate(unique_trial_types):
        trials_for_tt = all_lick_events[str(tt)]
        color = palette[i % len(palette)]
        for trial_in_type_idx, lick_times in sorted(trials_for_tt.items()):
            if lick_times:
                event_positions.append(list(lick_times))
                event_colors.append(color)
                y_labels.append(f"TT{tt} T{trial_in_type_idx + 1}")

    y_positions = list(range(len(event_positions)))

    plt.figure(figsize=(10, 5))
    if event_positions:
        plt.eventplot(event_positions, orientation="horizontal", colors=event_colors)

    plt.xlabel("Time (s)")
    plt.ylabel("Trial Index")
    plt.title(f"{mouse_id}_{date}")

    # Reference timings
    odor_start_time = 1.25
    odor_end_time = 1.75
    ymin = -0.5
    ymax = (max(y_positions) - 0.5) if y_positions else 0.5

    vline_times = [odor_start_time, odor_end_time,
                   odor_end_time + 0.25, odor_end_time + 1,
                   odor_end_time + 2.5, odor_end_time + 5.5]
    for t_v in vline_times:
        plt.vlines(t_v, ymin, ymax, color='purple', linestyle='-', linewidth=1, alpha=0.7)
    plt.axvspan(odor_start_time, odor_end_time, ymin=0, ymax=1, facecolor='purple', alpha=0.1)

    x_ticks_labels = ['odor', '0.75', '1.5', '3', '6']
    x_tick_positions = [odor_start_time,
                        odor_end_time + 0.25,
                        odor_end_time + 1,
                        odor_end_time + 2.5,
                        odor_end_time + 5.5]
    plt.xlim([0, 15])
    plt.xticks(x_tick_positions, x_ticks_labels)
    plt.grid(True, axis='x', linestyle='--')
    plt.gca().invert_yaxis()

    fig_root_dir = Path(fig_root_dir)
    fig_root_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_root_dir / f'{mouse_id}_{date}_raster.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()

    return fig_path


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    protocol = 'CombinedStimOdorTask'
    print('protocol: ' + protocol)
    mouse_id = "CC" + input("MouseID: CC")
    date = "2026" + input("Date (eg. 0126): 2026")

    # temp cache for long runs
    temp_data_folder = Path(r'C:\Users\carol\Github\lick_detection\tmp')
    temp_data_folder.mkdir(parents=True, exist_ok=True)
    temp_data_file = temp_data_folder / f'{mouse_id}_{date}_temp.pkl'

    lick_timestamps = None
    video_duration = None
    fps = None
    lick_frames_array = None
    selected_roi = None
    synced_gpio_file = None

    if temp_data_file.exists():
        print(f"Loading temporary data from {temp_data_file}")
        with open(temp_data_file, 'rb') as f:
            temp_data = pickle.load(f)
        lick_timestamps   = temp_data.get('lick_timestamps')
        video_duration    = temp_data.get('video_duration')
        fps               = temp_data.get('fps')
        synced_gpio_file  = temp_data.get('synced_gpio_file')
        selected_roi      = temp_data.get('selected_roi')
        lick_frames_array = temp_data.get('lick_frames_array')
        start_frame_for_playback = temp_data.get('start_frame_for_playback', 0)
    else:
        root_bpod_dir  = Path(r"\\140.247.90.110\homes2\Carol\BpodData")
        session_dir    = root_bpod_dir / mouse_id / protocol / "Session Data"
        pattern        = f"{mouse_id}_{protocol}_{date}_*.mat"
        matches        = list(session_dir.glob(pattern))
        bpod_data_path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
        if bpod_data_path is None:
            print("No session file found")
            sys.exit(1)

        root_video_dir = Path(r"\\140.247.90.110\homes2\Carol\VideoData10s")

        # Safe 'next' with default None
        video_path = next((p for p in root_video_dir.glob(f"{mouse_id}_{date}_*_cam1.avi")), None)
        if video_path is None:
            print("No video found")
            sys.exit(1)

        gpio_file_path = next((p for p in root_video_dir.glob(f"{mouse_id}_{date}_*_gpio1.csv")), None)
        if gpio_file_path is None:
            print("No GPIO file found")
            sys.exit(1)

        # Build synced dataframe
        synced_gpio_file = sync_data.sync_bpod_video(
            gpio_file_path=gpio_file_path,
            bpod_data_path=bpod_data_path
        )

        # ROI selection (from the middle) + remember the frame index used
        enter_roi_manually = input("Enter roi manually? (y/n): ").lower()
        if enter_roi_manually == 'y':
            roi_input = input("Enter ROI as (x, y, w, h): ")
            selected_roi = tuple(map(int, roi_input.strip("()").split(",")))
            start_frame_for_playback = 0
        else:
            (selected_roi, start_frame_for_playback) = select_roi(
                video_path,
                source_frame="middle",
                skip_head_seconds=0,  # set to e.g. 10.0 if you know the first 10 s aren't relevant
                skip_tail_seconds=0,
            )

        if not selected_roi:
            print("No ROI selected; exiting.")
            sys.exit(1)

        print(f"Selected ROI: {selected_roi} (start playback at frame {start_frame_for_playback})")

        # Show annotated playback?
        show_video = input("Video playback with lick detection? (y/n): ").lower()
        show_video_with_licks = (show_video == 'y')

        playback_speed = 0.5
        if show_video_with_licks:
            try:
                speed_input = float(input("Enter playback speed: "))
                if speed_input > 0:
                    playback_speed = speed_input
                else:
                    print("Playback speed must be positive. Using default (0.5).")
            except ValueError:
                print("Invalid input for playback speed. Using default (0.5).")

        # Save the annotated video to disk
        out_dir = Path(r"C:\Users\carol\Github\lick_detection\out")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{mouse_id}_{date}_annotated.mp4"

        # Run detection: start displaying from the same frame as ROI selection,
        # but save the full annotated session to disk.
        lick_timestamps, lick_frames_array, video_duration, fps = detect_licks(
            str(video_path),
            selected_roi,
            show_video_with_licks=show_video_with_licks,
            playback_speed=playback_speed,
            synced_gpio_file=synced_gpio_file,
            save_video_path= str(save_path), # [change here]
            save_codec="mp4v",
            save_fps=30.0,
            start_at_frame=start_frame_for_playback,   # start on-screen in the middle
            save_full_video=False,                      # save entire video annotated
            skip_head_seconds=0,                     # disable detection before this time (seconds) [change here]
            skip_tail_seconds=0,                     # disable detection near the end (seconds) [change here]
        )

        # Save temp cache unless you were watching the playback
        if show_video != 'y':
            with open(temp_data_file, 'wb') as f:
                pickle.dump({
                    'lick_timestamps': lick_timestamps,
                    'video_duration' : video_duration,
                    'fps'            : fps,
                    'synced_gpio_file': synced_gpio_file,
                    'selected_roi'   : selected_roi,
                    'lick_frames_array': lick_frames_array,
                    'start_frame_for_playback': start_frame_for_playback,
                }, f)
            print(f"Data saved to {temp_data_file}")

    # Attach lick flags and plot
    if lick_frames_array is None:
        print("Error: lick_frames_array is missing; cannot plot.")
        sys.exit(1)

    synced_gpio_file = synced_gpio_file.copy()
    synced_gpio_file['lickframe'] = lick_frames_array

    fig_path = plot_lick_raster(synced_gpio_file, fps, mouse_id, date)
    print(f"Saved raster: {fig_path}")



# modify [change here]