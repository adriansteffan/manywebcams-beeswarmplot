import cv2
import json

import shutil
import os
import sys
from os import listdir
from os.path import isfile, join
import subprocess
import pandas as pd
import statistics
from io import StringIO


# information is not present in the data output, fix this at some point in time
STIMULUS_ASPECT_RATIO = 4.0/3.0

# sampling rate parameters in Hz
## sampling rate that all data gets resampled to - for visualization purposes only
RESAMPLE_SAMPLING_RATE = 15

LAB_DATA_DIR = "./lab_data"
MEDIA_DIR = "./videos"
OUTPUT_DIR = "./output"

DATA_CSV = "./transformed_data_resampled.csv"
POSTERIOR_EXCLUSION_TXT = "participants_excluded_after_pre.txt"


def translate_coordinates(video_aspect_ratio, win_height, win_width, vid_height, vid_width, winX, winY):
    """translate the output coordinates of the eye-tracker onto the stimulus video"""
    if win_width/win_height > video_aspect_ratio:  # full height video
        vid_on_screen_width = win_height*video_aspect_ratio
        outside = False

        if winX < (win_width - vid_on_screen_width)/2 or winX > ((win_width - vid_on_screen_width)/2 + vid_on_screen_width):
            outside = True
        # scale x
        vidX = ((winX - (win_width - vid_on_screen_width)/2) / vid_on_screen_width) * vid_width
        # scale y
        vidY = (winY/win_height)*vid_height
        return int(vidX), int(vidY), outside
    else:  # full width video - not used in current study
        return None, None, True


def create_beeswarm(media_name, resampled_df, name_filter, show_sd_circle):

    """ create a beeswarm plot for a stimulus given a df with the resampled gaze data"""

    pre_path = OUTPUT_DIR+"/"+media_name+"_beeswarm_tobedeleted_" + name_filter + ".mp4"
    final_path = OUTPUT_DIR + "/" + media_name + "_beeswarm_" + ("sd_" if show_sd_circle else "")+ name_filter + ".mp4"

    # add frame counter to video
    p1 = subprocess.Popen(['ffmpeg',
                     '-y',
                     '-i',
                     MEDIA_DIR+"/"+media_name+".mp4",
                     '-vf',
                     "drawtext=fontfile=Arial.ttf: text='%{frame_num} / %{pts}': start_number=1: x=(w-tw)/2: y=h-lh: fontcolor=black: fontsize=(h/20): box=1: boxcolor=white: boxborderw=5",
                     "-c:a",
                     "copy",
                     "-c:v",
                     "libx264",
                     "-crf",
                     "23",
                     pre_path,
                     ])
    p1.wait()

    #filter dataframe by name_filter and trial
    clean_df = resampled_df[(resampled_df['stimulus'] == media_name) & (resampled_df['subid'].str.contains(name_filter))]

    # tag the video with eye tracking data
    video = cv2.VideoCapture(pre_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    vid_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (vid_width, vid_height), True)
    success, frame = video.read()
    index = 1
    timestep = 1000 / RESAMPLE_SAMPLING_RATE
    t = 0

    while success:

        relevant_rows = clean_df[clean_df['t'] == int(t)]

        if t <= (index/fps)*1000:
            t += timestep

        x_values = []
        y_values = []
        for i, row in relevant_rows.iterrows():

            x, y, outside = translate_coordinates(STIMULUS_ASPECT_RATIO,
                                         row['windowHeight'],
                                         row['windowWidth'],
                                         vid_height,
                                         vid_width,
                                         row['x'],
                                         row['y']
                                         )

            x_values.append(x)
            y_values.append(y)

            if not outside:
                cv2.circle(frame, (x, y), radius=10, color=(255, 0, 0), thickness=-1)
        try:
            cv2.circle(frame, (int(statistics.mean(x_values)), int(statistics.mean(y_values))), radius=15, color=(0, 0, 255), thickness=-1)
        except Exception:
            pass

        try:
            if show_sd_circle:
                cv2.ellipse(frame,
                            (int(statistics.mean(x_values)), int(statistics.mean(y_values))),
                            (int(statistics.stdev(x_values)), int(statistics.stdev(y_values))), 0., 0., 360, (255, 255, 255), thickness=3)
        except Exception:
            pass

        #cv2.imshow(media_name, frame)
        cv2.waitKey(int(1000 / int(fps)))
        video_writer.write(frame)
        success, frame = video.read()
        index += 1

    video.release()
    os.remove(pre_path)


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if os.path.exists(LAB_DATA_DIR):
    df = pd.DataFrame()
    posterior_exclusions = []

    print("Joining dataframes and exclusion lists...")
    for lab in os.listdir(LAB_DATA_DIR):
        if not os.path.isdir(os.path.join(LAB_DATA_DIR, lab)):
            continue

        data = pd.read_csv(os.path.join(LAB_DATA_DIR, lab, DATA_CSV))
        df = df.append(data, ignore_index=True)

        exclusion_path = os.path.join(LAB_DATA_DIR, lab, POSTERIOR_EXCLUSION_TXT)
        if os.path.exists(exclusion_path):
            with open(exclusion_path) as excl_file:
                posterior_exclusions += [line.rstrip() for line in excl_file]

    df.to_csv(os.path.join(OUTPUT_DIR, DATA_CSV), encoding='utf-8')
    with open(os.path.join(OUTPUT_DIR, POSTERIOR_EXCLUSION_TXT), 'w') as f:
        for participant in posterior_exclusions:
            f.write(f"{participant}\n")

    print("Dataframes and exclusion lists joined!")

else:
    df = pd.read_csv(DATA_CSV)
    if os.path.exists(POSTERIOR_EXCLUSION_TXT):
        with open(POSTERIOR_EXCLUSION_TXT) as file:
            posterior_exclusions = [line.rstrip() for line in file]
    else:
        posterior_exclusions = []

# posterior exclusions
df['subid'] = df['subid'].map(lambda x: str(x)[:-2])
df = df[~df['subid'].isin(posterior_exclusions)]

videos = ["FAM_LL", "FAM_LR", "FAM_RL", "FAM_RR"]

for v in videos:
    create_beeswarm(v, df, "", True)
    create_beeswarm(v, df, "", False)



