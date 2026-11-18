Expanded Classroom Dataset Usage Instructions
1 All original images from the selected classes are included; no duplicates were added to inflate the sample count. After extraction, run python check_data.py to verify the data. Only NumPy is required; no internet connection is needed.
2 dataset.npz contains x_train, y_train, x_test, and y_test. Images use uint8 with layout N,C,H,W; labels use int64 and start at 0.
3 Reserve one fifth of each class in train, rounded down, for validation, with the random seed fixed at 42; use the rest for training. See dataset_info.json for exact sizes and per-class distributions.
4 Convert images to float32 and divide by 255. Use N,C,H,W for PyTorch and convert to N,H,W,C for Keras; feed the data to the model in batches.
5 labels.txt provides the label mapping; source.txt lists the sources. The original number of classes and task difficulty are retained; only the sample count is increased. The original gesture and flower datasets have limited sizes, and all samples from the selected classes have already been included.
6 This package does not include training solutions. Students must implement the model and training loop themselves. The test set must not be used for model training.
