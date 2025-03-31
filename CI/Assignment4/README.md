# Non-Line-of-Sight (NLOS) Reconstruction Assignment

This repository contains code for NLOS reconstruction using three different methods:
1. Naive backprojection
2. Confocal backprojection
3. Attenuation-compensated backprojection

## How to Run the Code

The main file `main.m` is divided into three parts, each corresponding to a different reconstruction method. By default, only Part 1 is uncommented and will run.

### Setup

1. Make sure your data folder contains the necessary `.mat` files
2. Check that the paths in `main.m` point to the correct locations:
   ```matlab
   root_data_path = '/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CI/Assignment4/data/';
   root_output_path = '/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CI/Assignment4/output/';
   ```

### Running Different Parts

#### Part 1: Naive Backprojection
To run the naive backprojection analysis:
2. Comment out Parts 2 and 3 if not needed
3. Important: When processing multiple datasets, keep `volumeViewer()` calls commented to avoid opening too many viewers
4. Run `main.m`

To view individual reconstruction results:
1. Modify the dataset selection to focus on one dataset:
   ```matlab
   % data_set_list = {dataset_name1, dataset_name2, dataset_name3, dataset_name4, dataset_name5};
   data_set_list = {dataset_name4};  % Select individually for taking screenshots
   ```
2. Uncomment the volumeViewer calls to visualize the results:
   ```matlab
   volumeViewer(reconstructed_volume);
   volumeViewer(filtered_volume);
   ```
Part 1 includes automated performance analysis, generating:
1. A CSV file with timing information
2. A chart comparing reconstruction times across datasets and parameters


#### Part 2: Confocal Reconstruction
To run the confocal reconstruction:
1. Uncomment the code section between `% Part 2 %` and `% Part 3 %`
2. Comment out Parts 1 and 3 if not needed
3. Run `main.m`
4. The volumeViewer is already set up to display the results

#### Part 3: Attenuation-Compensated Reconstruction
This part is uncommented by default. To run only this part:
1. Ensure Parts 1 and 2 remain commented out
2. Run `main.m`
3. The volumeViewer will display reconstructed volumes automatically

## Output

Each reconstruction method produces:
1. Reconstructed volumes saved as `.mat` files in the output directory
2. Performance metrics (for Part 1)
3. Visual representation of the volumes through volumeViewer
