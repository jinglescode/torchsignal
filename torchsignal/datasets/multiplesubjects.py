import numpy as np

from torchsignal.datasets import OPENBMI
from torchsignal.filter.channels import pick_channels
from torchsignal.filter.butterworth import butter_bandpass_filter
from torchsignal.transform.segment import segment_signal
from torchsignal.datasets.utils import onehot_targets
from torchsignal.datasets.dataset import PyTorchDataset
from torchsignal.datasets.utils import train_test_split


def _load_multiple(root, dataset: PyTorchDataset, subject_ids: [], sessions: [], verbose: bool = False) -> None:
    data_by_subjects = {}

    for subject_id in subject_ids:
        print('Load subject:', subject_id)
        subject_data = None
        subject_target = None

        for session in sessions:
            subject_dataset = dataset(root=root, subject_id=subject_id, session=session)

            if subject_data is None: # if its session #1, will be None
                subject_data = np.zeros((0, subject_dataset.data.shape[1], subject_dataset.data.shape[2]))
                subject_target = np.zeros((0, ))

            subject_data = np.concatenate((subject_data, subject_dataset.data))
            subject_target = np.concatenate((subject_target, subject_dataset.targets))

        subject_dataset_new = PyTorchDataset(data=subject_data, targets=subject_target)
        subject_dataset_new.set_channel_names(subject_dataset.channel_names)
        data_by_subjects[subject_id] = subject_dataset_new
    
    return data_by_subjects


def _process_data(data_by_subjects, selected_channels, segment_config):

    for subject_id in list(data_by_subjects.keys()):
        subject_dataset = data_by_subjects[subject_id]

        subject_data = subject_dataset.data
        
        # filter channels
        if selected_channels is not None:
            subject_data = pick_channels(
                data=subject_data, 
                channel_names=subject_dataset.channel_names, 
                selected_channels=selected_channels
            )

        # segment signal
        if segment_config is not None:
            subject_data = segment_signal(
                signal=subject_data,
                window_len=segment_config['window_len'],
                shift_len=segment_config['shift_len'],
                sample_rate=segment_config['sample_rate'],
                add_segment_axis=segment_config['add_segment_axis'],
            )

        subject_data_full = np.zeros((subject_data.shape[0], subject_data.shape[1], subject_data.shape[3]))

        for trial in range(0, subject_data_full.shape[0]):
            for channel in range(0, subject_data_full.shape[1]):
                subject_data_full[trial, channel, :] = subject_data[trial, channel, 0, :]

        subject_data = subject_data_full

        # filter by bandpass
        subject_data = butter_bandpass_filter(subject_data, lowcut=6, highcut=15, sample_rate=1000, order=6)

        subject_dataset.set_data_targets(data=subject_data)




def _train_test_dataset(data_by_subjects):
    train_dataset_by_subjects = {}
    test_dataset_by_subjects = {}

    for subject_id in list(data_by_subjects.keys()):
        train_dataset, test_dataset = train_test_split(data_by_subjects[subject_id].data, data_by_subjects[subject_id].targets)
        
        train_dataset_by_subjects[subject_id] = train_dataset
        test_dataset_by_subjects[subject_id] = test_dataset
    
    return train_dataset_by_subjects, test_dataset_by_subjects



class MultipleSubjects():

    def __init__(self, 
        dataset: PyTorchDataset, 
        root: str, 
        subject_ids: [], 
        sessions: [], 
        selected_channels: [] = None,
        segment_config: {} = None,
        verbose: bool = False, 
    ) -> None:

        self.data_by_subjects = _load_multiple(
            root=root, 
            dataset=dataset, 
            subject_ids=subject_ids, 
            sessions=sessions
        )

        _process_data(
            data_by_subjects=self.data_by_subjects, 
            selected_channels=selected_channels,
            segment_config=segment_config,
        )

        self.train_dataset_by_subjects, self.val_dataset_by_subjects = _train_test_dataset(self.data_by_subjects)

    def leave_one_subject_out(self, selected_subject_id=1):

        assert selected_subject_id in self.data_by_subjects, "Must select subjects in dataset"

        # selected subject
        selected_subject_x = self.data_by_subjects[selected_subject_id].data
        selected_subject_y = self.data_by_subjects[selected_subject_id].targets
        test_dataset = PyTorchDataset(selected_subject_x, selected_subject_y)

        # the rest
        other_subjects_x_train = []
        other_subjects_y_train = []
        other_subjects_x_val = []
        other_subjects_y_val = []

        for subject_id in list(self.data_by_subjects.keys()):
            if subject_id != selected_subject_id:
                other_subjects_x_train.extend(self.train_dataset_by_subjects[subject_id].data)
                other_subjects_y_train.extend(self.train_dataset_by_subjects[subject_id].targets)

                other_subjects_x_val.extend(self.val_dataset_by_subjects[subject_id].data)
                other_subjects_y_val.extend(self.val_dataset_by_subjects[subject_id].targets)

        other_subjects_x_train = np.array(other_subjects_x_train)
        other_subjects_y_train = np.array(other_subjects_y_train)
        other_subjects_x_val = np.array(other_subjects_x_val)
        other_subjects_y_val = np.array(other_subjects_y_val)

        train_dataset = PyTorchDataset(other_subjects_x_train, other_subjects_y_train)
        val_dataset = PyTorchDataset(other_subjects_x_val, other_subjects_y_val)

        return train_dataset, val_dataset, test_dataset

