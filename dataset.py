import os
from pathlib import Path
import numpy as np
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """

    def __init__(self, directory, mode='train', clip_len=8,transforms=None):
        folder = Path(directory)  # get the directory of the specified split
        self.transforms = transforms
        self.clip_len = clip_len

        # the following three parameters are chosen as described in the paper section 4.1
        #self.resize_height = 30
        #self.resize_width = 60
        #self.crop_size = 80
        # obtain all the filenames of files inside all the class folders 
        # going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            if mode == 'train':
                for fname in os.listdir(os.path.join(folder, label))[:-10]:
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)
            elif mode == "val":
                for fname in os.listdir(os.path.join(folder, label))[-10:]:
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)  
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        
        with open("vertices_numbers.txt","r") as vr:
            numbers = ''.join(vr.readlines()).split("\n")
            self.numbers = np.array(list(map(int,numbers[:400])))
    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        # buffer = self.loadvideo(self.fnames[index])
        # if self.transforms:
        #     buffer = self.transforms(buffer)
        #buffer = self.crop(buffer, self.clip_len, self.crop_size)
        # buffer = self.normalize(buffer)
        buffer = np.fromfile(self.fnames[index],dtype = np.float32).reshape(100,20,61,3)
        # if self.transforms:
        #     buffer = self.transforms(buffer)
        buffer = buffer.transpose((3, 0, 1, 2))
        buffer = (buffer - np.mean(buffer))/np.std(buffer)

        return buffer, self.label_array[index]
        
        

    def loadvideo(self, fname, n_frame=20):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = []

        count = 0
        retaining = True
        sampling = np.linspace(0, frame_count-1, num=n_frame, dtype=int)
        # read in each frame, one at a time into the numpy buffer array
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if count in sampling:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = Image.fromarray(frame, 'RGB')
                buffer.append(frame)
            count += 1

        capture.release()
        return buffer 
    
    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[1] - clip_len)
        # randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[2] - crop_size)
        width_index = np.random.randint(buffer.shape[3] - crop_size)

        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[:, time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size]

        return buffer                

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed 
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel 
        # free to push to and edit this section to replace them if found. 
        new_buffer = []
        for b in buffer:
            new_buffer.append(np.array(b))
        buffer = np.stack(new_buffer)
        buffer = buffer.astype("float32")
        buffer = buffer.transpose((3, 0, 1, 2))
        buffer = (buffer - np.mean(buffer))/np.std(buffer)
        return buffer

    def __len__(self):
        return len(self.fnames)


class VideoDataset1M(VideoDataset):
    r"""Dataset that implements VideoDataset, and produces exactly 1M augmented
    training samples every epoch.
        
        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """
    def __init__(self, directory, mode='train', clip_len=8):
        # Initialize instance of original dataset class
        super(VideoDataset1M, self).__init__(directory, mode, clip_len)

    def __getitem__(self, index):
        # if we are to have 1M samples on every pass, we need to shuffle
        # the index to a number in the original range, or else we'll get an 
        # index error. This is a legitimate operation, as even with the same 
        # index being used multiple times, it'll be randomly cropped, and
        # be temporally jitterred differently on each pass, properly
        # augmenting the data. 
        
        index = np.random.randint(len(self.fnames))

        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)

        return buffer, self.label_array[index]    

    def __len__(self):
        return 1000000  # manually set the length to 1 million