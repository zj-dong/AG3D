echo Downloading pretrained model for deepfashion...
mkdir model
wget https://polybox.ethz.ch/index.php/s/zto1LoTIKygtX1h/download
mv ./download ./model/deep_fashion.pkl


echo Downloading pretrained models for ubcfashion...
wget https://polybox.ethz.ch/index.php/s/3ZI92nY4PqJDrI2/download
mv ./download ./model/ubc_fashion.pkl

echo Downloading preprocessed smpl sdf...
wget wget https://polybox.ethz.ch/index.php/s/Q5pyLvX4ECXEktR/download
mv ./download ./model/sdf_smpl.npy
