# Optimization and Comparison of Cloud Detection Algorithms

CloudMask Optimizer is a Python 3 package designed for the optimization, evaluation, and comparison of cloud detection algorithms 
for satellite imagery. The project integrates multiple deep learning models, including MFCNN, CloudFCN, CXN, and SEnSeI, 
to assess and enhance cloud-masking accuracy. The primary focus is on optimizing hyperparameters, improving model generalization, 
and benchmarking cloud detection performance across different datasets.

This repository supports cloud detection for Landsat 8 and Sentinel-2 imagery and allows quantitative and qualitative 
comparisons against traditional methods like Fmask.

## Installation

To install CloudMask Optimizer from source, follow these steps:
```
git clone https://github.com/aliFrancis/cloudFCN
cd cloudFCN
python setup.py install
```

## Dataset

This project utilizes multiple datasets for training, validation, and testing:

Biome dataset: Used for training and validation. Available at https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data. 

Landsat Collection 2 dataset: Used for testing. Available at https://www.sciencebase.gov/catalog/item/61015b2fd34ef8d7055d6395. 

Sentinel-2 dataset: Available at https://doi.org/10.5281/zenodo.4172871.

Before using the datasets, the raw images must be preprocessed. A cleaning script is used to convert them into a compatible 
format, normalize channels, and split scenes into smaller tiles. For example, to process the Biome dataset:

```
python cloudFCN/data/clean_biome_data.py  path/to/download  path/for/output  [-s splitsize] [-d downsample] ...
```
For Collection 2, use clean_data_set2.py, and for Sentinel-2, use sentinel_set.py.

To replicate our experimental setup, use the following parameters:

```
python cloudFCN/data/clean_biome_data.py  path/to/download  path/for/output  -s 384 -n True -t 0.8
```

This command generates 384×384 tiles with all available bands. Additionally, an extra ‘nodata’ band can be added to mark 
missing pixel values, ensuring they receive zero weight during training and validation.

The dataset cleaning script does not automatically split the data into training and validation sets. To do this, use the 
train_valid_test(_sentinel) function from data.Datasets, which has been integrated into the training pipeline.

Once the dataset is ready, the training process is handled by fit_model.py, which requires a JSON configuration file 
(fit_config.json). The script automates training and validation for a given experiment.

## Experiments

After preparing the dataset, you can reproduce the experiments from our research paper using the scripts in the experiments 
folder. These scripts train the MFCNN, CloudFCN, and CXN models. To train the SEnSeI model, use the corresponding script 
in the SEnSeI folder.

Although the results may vary slightly due to random model initialization, they should be close to the reported values.

To run the Biome dataset experiment:

```
python /path/to/experiments/fit_model.py /path/to/experiments/biome/multispectral/fit_config_1.json
```

Ensure that your configuration file has the correct dataset paths. If you're using a model_checkpoint_dir, make sure the 
directory exists.

After each training epoch, the script prints a table showing accuracy, commission, and omission errors for each validation 
set (e.g., Barren, Forest, etc.). This process can be time-consuming. To speed up training, modify the callback frequency 
in fit_model.py or increase steps_per_epoch in the configuration file.

## Results

To evaluate and analyze results, use the functions in the Fmask folder (compare_fmask_mask). This module compares 
Fmask-generated cloud masks with predictions from the trained model. Before running the comparison, ensure that the 
Fmask dataset is correctly prepared, and that images are split into patches.

The most important functions are located in compare_fmask_mask.py.

## Models and Metrics

Pretrained models are stored in the models folder. Additional metrics and evaluation tools can be found in the metric folder.

For further details, refer to the original paper or contact antonchajjka@gmail.com.


