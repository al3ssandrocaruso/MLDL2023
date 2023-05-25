import pickle
import pandas

ret_list = []
with open(file="./SUB_spectrogram/S04_spectrogram.pkl", mode='rb') as preprocessed_emg:
    for pre_index, pre_dict in pickle.load(preprocessed_emg).items():

        frames_sub_indexes = pre_dict["frames-sub_indexes-array"]
        label = pre_dict["label"]
        # print(pre_index)
        # print(pre_dict)
        # print()
        df_dict = {
            "uid": pre_index+1,
            "video_id": "S04_{:03d}".format(pre_index+1),
            "narration": label,
            "start_frame": frames_sub_indexes[0],
            "stop_frame": frames_sub_indexes[-1]
        }
        ret_list.append(df_dict)


df = pandas.DataFrame(ret_list)

# pandas.to_pickle(df, "../train_val_as/D1_test.pkl", protocol=4)

# WRITE to pickle
with open("../train_val_as/D1_test.pkl", "wb") as f:
    pickle.dump(df, f)

# read data
# with open(file="../train_val_as/D1_test.pkl", mode='rb') as f:
#     df = pandas.read_pickle(f)
#     print(df.head)


