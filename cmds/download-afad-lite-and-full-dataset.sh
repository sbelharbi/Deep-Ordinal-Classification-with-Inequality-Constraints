#!/usr/bin/env bash
# Script to download and extract the dataset: AFAD-LITE and FULL
# See: http://afad-dataset.github.io/

# cd to your folder where you want to save the data.
cd "$1"

# Download thes dataset.
echo "Downloading dataset AFAD-LITE ..."
git clone https://github.com/afad-dataset/tarball-lite.git
echo "Finished downloading AFAD-LITE."

cd tarball-lite
chmod +x restore.sh
./restore.sh

echo "Unzipping AFAD-Lite"
tar -xvf AFAD-Lite.tar.xz

echo "Finished extracting AFAD-Lite dataset."

cd ../

echo "Downloading dataset AFAD-FULL..."
git clone https://github.com/afad-dataset/tarball.git

echo "Finished downloading AFAD-Lite."

cd tarball
chmod +x restore.sh
./restore.sh

echo "Unzipping AFAD-Lite"
tar -xvf AFAD-Full.tar.xz

echo "Finished extracting AFAD-Full dataset."

