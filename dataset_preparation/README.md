# Dataset Preparation

## Processing Order
1. Extract features:
[`./script_video2feature.sh`](#feature-vector-frame-level-extraction)
2. Generate lists:
    * no official splits:
        1. [`./script_dataset2split.sh`](#dataset-split-generation)
        2. [`./script_dataset2list.sh`](#data-list-generation)
    * official splits exist (e.g. UCF, HMDB): [`./script_list2DA.sh`](#da-list-generation)

#### Notes
1. The options in the scripts have comments with the following types:
    * no comment: user can still change it, but NOT recommend (may need to change the code or have different experimental results)
    * comments with choices (e.g. `true | false`): can only choose from choices
    * comments as `depend on users`: totally depend on users (mostly related to data path)

---
### Feature vector (frame-level) extraction
`video2feature.py` loads the video dataset, and extract frame-level feature vectors, which are are needed for training and validation. A text file with all the category names is needed for labeled dataset.

Run `./script_video2feature.sh`.

There are two modes:
* labeled: `class_file=XXX/class_list_XXX.txt`. The features with the categories only listed in the `class_file` will generated.
* unlabeled: `class_file=none`: all the features with the category "unlabeled" will be generated.

Output folder structures:
```
data_path/
  RGB-Feature/
    VIDEO_0001/
      img_00001.t7
      img_00002.t7
      ...
    VIDEO_0002/
    ...
```

---
### Dataset split generation
`dataset2split.py` splits the whole dataset into two subsets according to the split ratio (randomly picking videos). Each subset has the same structure as shown above.

Run `./script_dataset2split.sh`.

Options:
* `input_type`: depend on the format of raw data
* `split_ratio`: ratio of training data
* `split_feat`: if the features are already generated, users can split the corresponding features as well.


---
### Data list generation
`video_dataset2list.py` generates the data list from a video dataset. The output text file will include all the selected categories showing {video_path, frame#, class}.

Run `./script_dataset2list.sh`.

Options:
* `random_each_video`: If `Y`, it means users randomly select video clips from each raw and unsegmented video. It `N`, it means users randomly select video clips from each category.
* `max_num`: maximal numbers of selected video clips for each unsegmented video or category (depend on `random_each_video`)
* `method_read`: the method of calculating frame # of each video
  * `video`: load from the raw video folder (slower, but more accurate)
  * `frame`: load from the feature folder
* `suffix`: add some string to the list file name

Output text file (format: [video_full_path frame# class]):
```
data_path/RGB-Feature/VIDEO_0001/ 100 0
data_path/RGB-Feature/VIDEO_0002/ 150 1
......
```

---
### DA list generation
If official splits exist (e.g. UCF, HMDB), `list_ucf_hmdb_full2DA.py` can separate the data list from the official splits with user chosen DA settings.

Run `./script_list2DA.sh`.
