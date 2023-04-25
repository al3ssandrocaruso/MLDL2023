import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np

from .utils import dense_sampling


class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3) -> this is the split we want to select in the folder train_val
        modalities: list(str, str, ...) -> group of modalities we are considering for the analysis
        mode: str (train, test/val) -> modality we want to use the dataset for
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """

        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        # creation of the pickle file name considering the split and the modality (e.g. D1 + _ + test + .pkl)
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        # get the pickle file location (path + name). The path must be inserted in dataset_conf.annotations_path
        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[
                    ["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB', sampling='uniform', clip_length=32, stride = 2):

        # ensure num_frames_per_clip has to be > clip_lenght
        start_frame = record.start_frame
        end_frame = record.end_frame
        central_min = start_frame + clip_length // 2
        central_max = end_frame - clip_length // 2

        # randomly sample N central points
        central_points = np.random.choice(range(central_min, central_max), size=self.num_clips, replace=False)

        # generate clips
        if clip_length % 2 == 0:
            clips = [(c - clip_length // 2, c + clip_length // 2 - 1) for c in central_points]
        else:
            clips = [(c - clip_length // 2, c + clip_length // 2) for c in central_points]

        output = []

        if sampling == 'uniform':
            for clip in clips:
                frame_indices = np.linspace(clip[0], clip[1], num=self.num_frames_per_clip, dtype=int)
                frames = [frame_indices[i] for i in range(self.num_frames_per_clip)]
                output.append(frames)

        elif sampling == 'dense':
            for clip in clips:
                output.append(dense_sampling(clip[0], clip[1], self.num_frames_per_clip, stride))

        return np.array(output).reshape(-1,1)

    def _get_val_indices(self, record, modality='RGB', sampling='uniform', clip_length=32, stride = 2):

        # ensure num_frames_per_clip has to be > clip_lenght
        start_frame = record.start_frame
        end_frame = record.end_frame
        central_min = start_frame + clip_length // 2
        central_max = end_frame - clip_length // 2

        # randomly sample N central points
        central_points = np.random.choice(range(central_min, central_max), size=self.num_clips, replace=False)

        # generate clips
        if clip_length % 2 == 0:
            clips = [(c - clip_length // 2, c + clip_length // 2 - 1) for c in central_points]
        else:
            clips = [(c - clip_length // 2, c + clip_length // 2) for c in central_points]

        output = []

        if sampling == 'uniform':
            for clip in clips:
                frame_indices = np.linspace(clip[0], clip[1], num=self.num_frames_per_clip, dtype=int)
                frames = [frame_indices[i] for i in range(self.num_frames_per_clip)]
                output.append(frames)

        elif sampling == 'dense':
            for clip in clips:
                output.append(dense_sampling(clip[0], clip[1], self.num_frames_per_clip, stride))

        return np.array(output).reshape(-1,1)

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]

        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
