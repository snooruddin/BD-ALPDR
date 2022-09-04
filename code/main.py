#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sheikh Nooruddin
"""

import lpdutils as lpd

#actual datapaths 480p
backPath = "./DATASET - SORTED/Back/"
backLeftPath = "./DATASET - SORTED/BackLeft/"
backRightPath = "./DATASET - SORTED/BackRight/"
frontPath = "./DATASET - SORTED/Front/"
frontLeftPath = "./DATASET - SORTED/FrontLeft/"
frontRightPath = "./DATASET - SORTED/FrontRight/"

#actual resized datapaths 480p
backResPath = "./DATASET - SORTED/Back/Resized/"
backLeftResPath = "./DATASET - SORTED/BackLeft/Resized/"
backRightResPath = "./DATASET - SORTED/BackRight/Resized/"
frontResPath = "./DATASET - SORTED/Front/Resized/"
frontLeftResPath = "./DATASET - SORTED/FrontLeft/Resized/"
frontRightResPath = "./DATASET - SORTED/FrontRight/Resized/"

#train datapaths
backTrainPath = "./train/bus and trucks/Bus and Trucks/Back/"
backLeftTrainPath = "./train/bus and trucks/Bus and Trucks/BackLeft/"
backRightTrainPath = "./train/bus and trucks/Bus and Trucks/BackRight/"
frontTrainPath = "./train/bus and trucks/Bus and Trucks/Front/"
frontLeftTrainPath = "./train/bus and trucks/Bus and Trucks/FrontLeft/"
frontRightTrainPath = "./train/bus and trucks/Bus and Trucks/FrontRight/"

#test datapaths
backTestPath = "./test/Bus and Trucks/Back/"
backLeftTestPath = "./test/Bus and Trucks/BackLeft/"
backRightTestPath = "./test/Bus and Trucks/BackRight/"
frontTestPath = "./test/Bus and Trucks/Front/"
frontLeftTestPath = "./test/Bus and Trucks/FrontLeft/"
frontRightTestPath = "./test/Bus and Trucks/FrontRight/"

#ground truth datapaths
gtBackTestPath = "./gt/Bus and Trucks/Back/"
gtBackLeftTestPath = "./gt/Bus and Trucks/BackLeft/"
gtBackRightTestPath = "./gt/Bus and Trucks/BackRight/"
gtFrontTestPath = "./gt/Bus and Trucks/Front/"
gtFrontLeftTestPath = "./gt/Bus and Trucks/FrontLeft/"
gtFrontRightTestPath = "./gt/Bus and Trucks/FrontRight/"

#test results datapath
backTestResultsPath = "./test/Bus and Trucks/Back/results/"
backLeftTestResultsPath = "./test/Bus and Trucks/BackLeft/results/"
backRightTestResultsPath = "./test/Bus and Trucks/BackRight/results/"
frontTestResultsPath = "./test/Bus and Trucks/Front/results/"
frontLeftTestResultsPath = "./test/Bus and Trucks/FrontLeft/results/"
frontRightTestResultsPath = "./test/Bus and Trucks/FrontRight/results/"

#extracted lp regions paths
backLpPath = "./lpregions/bus and trucks/Back/"
backLeftLpPath = "./lpregions/bus and trucks/BackLeft/"
backRightLpPath = "./lpregions/bus and trucks/BackRight/"
frontLpPath = "./lpregions/bus and trucks/Front/"
frontLeftLpPath = "./lpregions/bus and trucks/FrontLeft/"
frontRightLpPath = "./lpregions/bus and trucks/FrontRight/"

#extracted non lp regions paths
backNonLpPath = "./nonlpregions/bus and trucks/Back/"
backLeftNonLpPath = "./nonlpregions/bus and trucks/BackLeft/"
backRightNonLpPath = "./nonlpregions/bus and trucks/BackRight/"
frontNonLpPath = "./nonlpregions/bus and trucks/Front/"
frontLeftNonLpPath = "./nonlpregions/bus and trucks/FrontLeft/"
frontRightNonLpPath = "./nonlpregions/bus and trucks/FrontRight/"

#performing batch resize

lpd.batch_resize(backPath, backResPath, keepAspectRatio=True)
lpd.batch_resize(backLeftPath, backLeftResPath, keepAspectRatio=True)
lpd.batch_resize(backRightPath, backRightResPath, keepAspectRatio=True)
lpd.batch_resize(frontPath, frontResPath, keepAspectRatio=True)
lpd.batch_resize(frontLeftPath, frontLeftResPath, keepAspectRatio=True)
lpd.batch_resize(frontRightPath, frontRightResPath, keepAspectRatio=True)


#performing the train test splits of images.

lpd.train_test_image_split(backResPath)
lpd.train_test_image_split(backLeftResPath)
lpd.train_test_image_split(backRightResPath)
lpd.train_test_image_split(frontResPath)
lpd.train_test_image_split(frontLeftResPath)
lpd.train_test_image_split(frontRightResPath)


#performing annotization of images

lpd.single_region_extract_with_replace(backTrainPath, backLpPath, backNonLpPath)
lpd.single_region_extract_with_replace(backLeftTrainPath, backLeftLpPath, backLeftNonLpPath)
lpd.single_region_extract_with_replace(backRightTrainPath, backRightLpPath, backRightNonLpPath)
lpd.single_region_extract_with_replace(frontTrainPath, frontLpPath, frontNonLpPath)
lpd.single_region_extract_with_replace(frontLeftTrainPath, frontLeftLpPath, frontLeftNonLpPath)
lpd.single_region_extract_with_replace(frontRightTrainPath, frontRightLpPath, frontRightNonLpPath)


#get ground truths

lpd.single_region_extract(backTestPath, gtBackTestPath)
lpd.single_region_extract(backLeftTestPath, gtBackLeftTestPath)
lpd.single_region_extract(backRightTestPath, gtBackRightTestPath)
lpd.single_region_extract(frontTestPath, gtFrontTestPath)
lpd.single_region_extract(frontLeftTestPath, gtFrontLeftTestPath)
lpd.single_region_extract(frontRightTestPath, gtFrontRightTestPath)


#extract glcm features from lp regions

lpd.get_GLCM_features_RGB(backLpPath, 1, csvFileName="lp_glcm_output.csv")
lpd.get_GLCM_features_RGB(backLeftLpPath, 1, csvFileName="lp_glcm_output.csv")
lpd.get_GLCM_features_RGB(backRightLpPath, 1, csvFileName="lp_glcm_output.csv")
lpd.get_GLCM_features_RGB(frontLpPath, 1, csvFileName="lp_glcm_output.csv")
lpd.get_GLCM_features_RGB(frontLeftLpPath, 1, csvFileName="lp_glcm_output.csv")
lpd.get_GLCM_features_RGB(frontRightLpPath, 1, csvFileName="lp_glcm_output.csv")

#extract CHPOOL RGB features from lp regions

lpd.get_CHPOOL_features_RGB(backLpPath, 1, csvFileName="lp_chpool_output.csv")
lpd.get_CHPOOL_features_RGB(backLeftLpPath, 1, csvFileName="lp_chpool_output.csv")
lpd.get_CHPOOL_features_RGB(backRightLpPath, 1, csvFileName="lp_chpool_output.csv")
lpd.get_CHPOOL_features_RGB(frontLpPath, 1, csvFileName="lp_chpool_output.csv")
lpd.get_CHPOOL_features_RGB(frontLeftLpPath, 1, csvFileName="lp_chpool_output.csv")
lpd.get_CHPOOL_features_RGB(frontRightLpPath, 1, csvFileName="lp_chpool_output.csv")

#extract glcm features from non lp regions

lpd.get_GLCM_features_RGB(backNonLpPath, 0, csvFileName="nonlp_glcm_output.csv", stepSize = 10)
lpd.get_GLCM_features_RGB(backLeftNonLpPath, 0, csvFileName="nonlp_glcm_output.csv", stepSize = 10)
lpd.get_GLCM_features_RGB(backRightNonLpPath, 0, csvFileName="nonlp_glcm_output.csv", stepSize = 10)
lpd.get_GLCM_features_RGB(frontNonLpPath, 0, csvFileName="nonlp_glcm_output.csv", stepSize = 10)
lpd.get_GLCM_features_RGB(frontLeftNonLpPath, 0, csvFileName="nonlp_glcm_output.csv", stepSize = 10)
lpd.get_GLCM_features_RGB(frontRightNonLpPath, 0, csvFileName="nonlp_glcm_output.csv", stepSize = 10)

#extract CHPOOL RGB features from non lp regions

lpd.get_CHPOOL_features_RGB(backNonLpPath, 0, csvFileName="nonlp_chpool_output.csv", stepSize = 10)
lpd.get_CHPOOL_features_RGB(backLeftNonLpPath, 0, csvFileName="nonlp_chpool_output.csv", stepSize = 10)
lpd.get_CHPOOL_features_RGB(backRightNonLpPath, 0, csvFileName="nonlp_chpool_output.csv", stepSize = 10)
lpd.get_CHPOOL_features_RGB(frontNonLpPath, 0, csvFileName="nonlp_chpool_output.csv", stepSize = 10)
lpd.get_CHPOOL_features_RGB(frontLeftNonLpPath, 0, csvFileName="nonlp_chpool_output.csv", stepSize = 10)
lpd.get_CHPOOL_features_RGB(frontRightNonLpPath, 0, csvFileName="nonlp_chpool_output.csv", stepSize = 10)

#get shapes of all license plates
#sanity check
lpd.__getShapes__(backLpPath)
lpd.__getShapes__(backLeftLpPath)
lpd.__getShapes__(backRightLpPath)
lpd.__getShapes__(frontLpPath)
lpd.__getShapes__(frontLeftLpPath)
lpd.__getShapes__(frontRightLpPath)

# the models must be trained on the features at this point.
# the trained models will be used in the following steps.

#draw detected regions in test images

lpd.draw_probable_LPRegions_CHPOOL_RGB(backTestPath, "./", backTestResultsPath, stepSize=12, confidence=0.90)
lpd.draw_probable_LPRegions_CHPOOL_RGB(backLeftTestPath, "./", backLeftTestResultsPath, stepSize=12, confidence=0.90)
lpd.draw_probable_LPRegions_CHPOOL_RGB(backRightTestPath, "./", backRightTestResultsPath, stepSize=12, confidence=0.90)
lpd.draw_probable_LPRegions_CHPOOL_RGB(frontTestPath, "./", frontTestResultsPath, stepSize=12, confidence=0.90)
lpd.draw_probable_LPRegions_CHPOOL_RGB(frontLeftTestPath, "./", frontLeftTestResultsPath, stepSize=12, confidence=0.90)
lpd.draw_probable_LPRegions_CHPOOL_RGB(frontRightTestPath, "./", frontRightTestResultsPath, stepSize=12, confidence=0.90)

#calculate IoU metrics for results

lpd.compute_IoU("./IOUResult/backoutput.csv","./IOUResult/backgt.csv","./IOUResult/","./backresult.csv")
lpd.compute_IoU("./IOUResult/backleftoutput.csv","./IOUResult/backleftgt.csv","./IOUResult/","./backleftresult.csv")
lpd.compute_IoU("./IOUResult/backrightoutput.csv","./IOUResult/backrightgt.csv","./IOUResult/","./backrightresult.csv")
lpd.compute_IoU("./IOUResult/frontoutput.csv","./IOUResult/frontgt.csv","./IOUResult/","./frontresult.csv")
lpd.compute_IoU("./IOUResult/frontleftoutput.csv","./IOUResult/frontleftgt.csv","./IOUResult/","./frontleftresult.csv")
lpd.compute_IoU("./IOUResult/frontrightoutput.csv","./IOUResult/frontrightgt.csv","./IOUResult/","./frontrightresult.csv")

# show the results images

lpd.view_images(backTestResultsPath, 640, 480)
lpd.view_images(backLeftTestResultsPath, 640, 480)
lpd.view_images(backRightTestResultsPath, 640, 480)
lpd.view_images(frontTestResultsPath, 640, 480)
lpd.view_images(frontLeftTestResultsPath, 640, 480)
lpd.view_images(frontRightTestResultsPath, 640, 480)


