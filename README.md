This is a sleep analysis toolkit. It currently contains one project Snore detection.
But the current version assumes:
1) Every routine with signal as an input assumes data is a (t,1) array, not
    a path to file.
2) Any feature matrix has time/frame_index as 1st index so
    np.ndarray(shape=(num_of_frames,feat_dim))
3) Little sanity check of inputs, we assume ppl know what they are doing.
    Some sanity check by setting dtype=np.uint32 for nfft/flen etc.
4) Use dtype=np.uintXX when working with positive int values
    (nfft/indexing/flen...).
5) Avoid using float64 as it takes too much memory for very large signals and
    fine segmentation

TODO :
    implement resample
    implement gabor fb
    implement gammatone fb
    implement outlier detection (MAD)
    implement method for frequency to bins (nonlinear) and substitute in current functions?
    How to treat parameters such as time_to_frame(t...) or mel(f:float = 0) .... scalars/arrays?
        -> need to write some rudimentary checks on their values (i.e. if t < 0 then error)



INSTALL GUIDE
# Setup Virtual Environment
mkdir ~/VirtualEnv
python3 -m venv ~/VirtualEnv/snore_detect
source ~/VirtualEnv/snore_detect/bin/activate

# Install necessary packages
python3 -m pip install numpy
python3 -m pip install wheel
python3 -m pip install pyedflib
python3 -m pip install torch
python3 -m pip install torchvision
python3 -m pip install scipy


# Download the toolkit from GitHub


# Link vEnv with the project
ln -s ~/VirtualEnv/snore_detect ~/Projects/Snore/snore_detector/snore_detect

# Link toolkit with the project
ln -s ~/Projects/Snore/sleepat ~/Projects/Snore/snore_detector/sleepat
