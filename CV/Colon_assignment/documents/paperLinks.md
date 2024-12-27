# Endomapper paper:
https://www.nature.com/articles/s41597-023-02564-7

# Endomapper repo
https://github.com/Endomapper

## ORB-SLAM validation from endomapper in C++
https://github.com/endomapper/EM_Dataset-ORBSLAM3Validation

# Colmap repo
https://colmap.github.io/index.html

# Light glue repo
https://github.com/cvg/LightGlue

# Super glue (to use COLMAP)
https://github.com/magicleap/SuperGluePretrainedNetwork

# Commands
./match_pairs.py --input_pairs assets/colon_pairs_superglue.txt  --input_dir ../../data/Seq_027/ --output_dir colon_matches/ --viz --fast_viz

./match_pairs.py --input_pairs assets/colon_pairs_superglue.txt --input_dir ../../data/Seq_027/ --output_dir colon_matches/ --resize -1 --max_keypoints -1 --keypoint_threshold 0.003 --viz --fastviz

./match_pairs.py --input_pairs assets/colon_pairs_superglue.txt --input_dir ../../data/Seq_027/ --output_dir colon_matches/ --resize -1 --max_keypoints 2048 --keypoint_threshold 0.004 --viz --fastviz

python ./manipulate_npz.py
