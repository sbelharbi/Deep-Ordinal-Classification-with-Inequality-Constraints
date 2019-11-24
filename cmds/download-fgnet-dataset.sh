#!/usr/bin/env bash
# Script to download and extract the dataset: FGNET
# website:
#https://yanweifu.github.io/FG_NET_data/index.html
#
#
#FG-NET dataset by Yanwei Fu
#
#PhD student, Vision Group
#
#EECS, Queen Mary, University of London
#
#This is the FG-NET data. Obviously, the original FG-NET website does not
# provide this data any more. I provide them in my homepage. Cory (Ke Chen)
# gave me this data which is used in his paper
#[1] Fu, Yanwei; Hospedales, T.; Xiang, T.; Gong, S; Yao. Y:Interestingness
# Prediction by Robust Learning to Rank, (ECCV 2014). Paper
#
#bib: @INPROCEEDINGS{ranking2014ECCV,
#author = { Yanwei Fu and Timothy M. Hospedales and Tao Xiang and Yuan Yao and
# Shaogang Gong},
#title = {Interestingness Prediction by Robust Learning to Rank},
#booktitle = {ECCV},
#year = {2014}
#}
#[2] Ke Chen, Shaogang Gong, Tao Xiang, Chen Chang Loy, ``Cumulative attribute
# space for age and crowd density estimation,'' in IEEE Conference on Computer
# Vision and Pattern Recognition (CVPR), 2013
#
#
#
#To download all data:
#
#http://yanweifu.github.io/FG_NET_data/FGNET.zip
#
#The data explanations: (available if you start with
# http://www.eecs.qmul.ac.uk/~yf300/FG_NET_data/)
#./images folder: all human face images. The groundtruth is used to name each
# image. For example, 078A11.JPG, means that this is the No.'78' person's image
# when he/she was 11 years old. 'A' is short for Age.
#
#./points folder: this is the 68 manual annotated points for each image in
# ./images folder.
#The annotated data is of much higher quality than another dataset e.g. MORPH
# (saved in /export/beware/thumper/yf300/Age_estimation_org_data_backup/
# ageEstimation/MOPRH). However, MORPH is much bigger dataset than FG-NET.
#
#./feature_generation_tools: this is the tool to generate the features.
#./feature_generation_tools/how-to-use-it: tutorial of how to use the tools.
#./age50_10_round.mat is the 10 rounds of data used in my work [1].
#Normally, you should firstly split the training/testing data by yourself.
# And generate the low-level feature for training/testing data respectively.
# For each split, the training/testing features are not the same. Because
# the process of generating training features is also needed to refer the
# annotations of testing features.
#
#
#There is another very good tutorial and matlab labelling tool for AAM/ASM.
# You can download it from:
#http://yanweifu.github.io/FG_NET_data/AAM_verygood.rar
#But some of them were written in Chinese.
#PS: If any further questions, please email me: y.fu@qmul.ac.uk (and CC
# to ke.chen@tut.fi).
#Yanwei Fu, Aug. 5th, 2014
#Yanwei Fu, Aug. 22nd, 2014 --updated

# cd to your folder where you want to save the data.
cd "$1"

# Download ths dataset.
echo "Downloading dataset FGNET ..."
wget -O FGNET.zip http://yanweifu.github.io/FG_NET_data/FGNET.zip

echo "Finished downloading Oxford-flowers-102 dataset."

echo "Extracting files ..."

unzip FGNET.zip
ls


echo "Finished extracting Oxford-flowers-102 dataset."
