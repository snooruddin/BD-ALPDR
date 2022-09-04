#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sheikh Nooruddin
"""

import os
import random
from shutil import copy


import cv2 as cv
import numpy as np
import pandas as pd
import seaborn as sns

from skimage.feature import greycomatrix, greycoprops
from skimage.measure import block_reduce
import joblib

def single_region_extract(inputDirectoryPath, outputDirectoryPath, prefix="crop_", csvFileName="sre_output.csv"):
    """
    Function that:
        extracts image regions using supervision
        saves corordinates in a csv file.

    Parameters
    ----------
    inputDirectoryPath : String
        Path to the directory holding the images.

    outputDirectoryPath : String
        Path where the cropped lp and csv file is saved.

    prefix : String -> "crop_"
        Prefix of the output images.

    csvFileName: String -> sre_output.csv
        Name of the output csv file. Default: "sre_output.csv".
    ----------

    Usage
    -----
    >> single_region_extract("./train/", "./", "hello.csv")

    """

    reactangle = []

    fileNamess = os.listdir(inputDirectoryPath)

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    def region_selection(event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv.EVENT_LBUTTONDOWN:

            reactangle.append(x)
            reactangle.append(y)
            print(reactangle)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished

            reactangle.append(x)
            reactangle.append(y)

            # draw a rectangle around the region of interest
            print(reactangle)
            cv.rectangle(img, (reactangle[0], reactangle[1]), (reactangle[2], reactangle[3]), (0, 255, 0), 2)
            cv.imshow("image", img)
            # reactangle_copy = reactangle.copy()

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            # denotes top left to downright motion
            acrossFlag = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            clone = img.copy()
            cv.namedWindow("image")
            cv.setMouseCallback("image", region_selection)

            # keep looping until the 'q' key is pressed
            while True:

                # display the image and wait for a input from keyboard
                cv.imshow("image", img)
                key = cv.waitKey(1) & 0xFF

                # if 'r' key is pressed, the window is reset and
                if key == ord("r"):
                    img = clone.copy()
                    reactangle = []
                    break

                # if the space key is pressed, break from the loop
                elif key == ord(" "):
                    print(reactangle)
                    break

                elif key == ord("n"):
                    reactangle = []
                    break

            if len(reactangle) == 4:

                # cropping the part of the image and showing that
                if reactangle[1] < reactangle[3] and reactangle[0] < reactangle[2]:
                    # for left to right across
                    crop_img = clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :].copy()
                else:
                    # for right to left across
                    crop_img = clone[reactangle[1]:reactangle[3], reactangle[2]:reactangle[0], :].copy()
                    acrossFlag = 1

                crop_img_rsz = cv.resize(crop_img, (512, 512))
                cv.imshow("crop_img", crop_img)
                cv.imshow("crop resized img", crop_img_rsz)

                # saving the cropped image in output directory
                outFileName = outputDirectoryPath + prefix + file
                print("[ CROP: " + outFileName + " ]", end="\n\n\n")
                cv.imwrite(outFileName, crop_img)

                if acrossFlag:
                    reactangles = [min(reactangle[0], reactangle[2]), min(reactangle[1], reactangle[3]),
                                   max(reactangle[0], reactangle[2]), max(reactangle[1], reactangle[3])]
                    row = np.array([np.append(reactangles, file)])
                else:
                    row = np.array([np.append(reactangle, file)])

                # saving reactangle values as rows in output csv file
                # row = np.array([np.append(reactangle, file)])
                np.savetxt(f, row, delimiter=',', fmt='%s')

                # resetting reactangle for next iteration
                reactangle = []

                cv.waitKey(0)

                # close all open windows
            cv.destroyAllWindows()
            cv.waitKey(1)

            # np.savetxt(f, row, delimiter=",")

            # index += 1

            # print("[ " + str(index) + ". " + fullFilePath + "]")

    f.close()


def single_region_extract_with_replace(inputDirectoryPath, outputDirectoryPath1, outputDirectoryPath2, prefix="crop_",
                                   csvFileName="srer_output.csv"):
    """
    Function that:
        extracts image regions using supervision
        saves corordinates in a csv file and rep
        replaces cropped region with white.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the directory holding the images.

    outputDirectoryPath1: String
        Path where the cropped lp and csv file is saved.

    outputDirectoryPath1: String
        Path where the cropped lp and csv file is saved.

    prefix: String -> "crop_"
        Prefix of the cropped images.

    csvFileName: String -> sre_output.csv
        Name of the output csv file. Default: "sre_output.csv".
    ----------

    Usage
    -----
    >> single_region_extract("./train/", "./", "baby.csv")

    """

    reactangle = []

    fileNamess = os.listdir(inputDirectoryPath)

    os.makedirs(os.path.dirname(outputDirectoryPath1), exist_ok=True)
    outputCSVFile = outputDirectoryPath1 + csvFileName

    os.makedirs(os.path.dirname(outputDirectoryPath2), exist_ok=True)

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    def region_selection(event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv.EVENT_LBUTTONDOWN:

            reactangle.append(x)
            reactangle.append(y)
            print(reactangle)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished

            reactangle.append(x)
            reactangle.append(y)

            # draw a rectangle around the region of interest
            print(reactangle)
            cv.rectangle(img, (reactangle[0], reactangle[1]), (reactangle[2], reactangle[3]), (0, 255, 0), 2)
            cv.imshow("image", img)
            # reactangle_copy = reactangle.copy()

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            # denotes top left to downright motion
            acrossFlag = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            clone = img.copy()
            cv.namedWindow("image")
            cv.setMouseCallback("image", region_selection)

            # keep looping until the 'q' key is pressed
            while True:

                # display the image and wait for a input from keyboard
                cv.imshow("image", img)
                key = cv.waitKey(1) & 0xFF

                # if 'r' key is pressed, the window is reset and
                if key == ord("r"):
                    img = clone.copy()
                    reactangle = []
                    break

                # if the space key is pressed, break from the loop
                elif key == ord(" "):
                    print(reactangle)
                    break

                elif key == ord("n"):
                    reactangle = []
                    break

            if len(reactangle) == 4:

                # cropping the part of the image and showing that
                # copy to avoid reference issues

                if reactangle[1] < reactangle[3] and reactangle[0] < reactangle[2]:
                    # for left to right across
                    crop_img = clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :].copy()
                else:
                    # for right to left across
                    crop_img = clone[reactangle[1]:reactangle[3], reactangle[2]:reactangle[0], :].copy()
                    acrossFlag = 1

                # crop_img = clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :].copy()
                crop_img_rsz = cv.resize(crop_img, (512, 512))
                cv.imshow("crop_img", crop_img)
                cv.imshow("crop resized img", crop_img_rsz)

                # replacing the cropped part with white pixels
                if acrossFlag:
                    clone[reactangle[1]:reactangle[3], reactangle[2]:reactangle[0], :] = 255
                    reactangles = [min(reactangle[0], reactangle[2]), min(reactangle[1], reactangle[3]),
                                   max(reactangle[0], reactangle[2]), max(reactangle[1], reactangle[3])]
                    row = np.array([np.append(reactangles, file)])
                else:
                    clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :] = 255
                    row = np.array([np.append(reactangle, file)])

                # saving the cropped image in output directory
                outFileName = outputDirectoryPath1 + prefix + file
                print("[ CROP: " + outFileName + " ]", end="\n\n\n")
                cv.imwrite(outFileName, crop_img)

                # saving the replaced image in the second ouput directory
                outFileName = outputDirectoryPath2 + prefix + file
                print("[ REPLACE: " + outFileName + " ]", end="\n\n\n")
                cv.imwrite(outFileName, clone)

                # saving reactangle values as rows in output csv file
                # row = np.array([np.append(reactangle, file)])

                np.savetxt(f, row, delimiter=',', fmt='%s')

                # resetting reactangle for next iteration
                reactangle = []

                cv.waitKey(0)

                # close all open windows
            cv.destroyAllWindows()
            cv.waitKey(1)

            # np.savetxt(f, row, delimiter=",")

            # index += 1

            # print("[ " + str(index) + ". " + fullFilePath + "]")

    f.close()


def multi_region_extract_with_replace(inputDirectoryPath, outputDirectoryPath1, outputDirectoryPath2, prefix="crop_",
                                  csvFileName="mrer_output.csv"):
    """
    Function that:
        extracts image regions using supervision
        saves corordinates in a csv file and rep
        replaces cropped region with white.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the directory holding the images.

    outputDirectoryPath1: String
        Path where the cropped lp and csv file is saved.

    outputDirectoryPath1: String
        Path where the cropped lp and csv file is saved.

    prefix: String -> "crop_"
        Prefix of the cropped images.

    csvFileName: String -> sre_output.csv
        Name of the output csv file. Default: "sre_output.csv".
    ----------

    Usage
    -----
    >> single_region_extract("./train/", "./", "baby.csv")

    """

    reactangle = []

    fileNamess = os.listdir(inputDirectoryPath)

    os.makedirs(os.path.dirname(outputDirectoryPath1), exist_ok=True)
    outputCSVFile = outputDirectoryPath1 + csvFileName

    os.makedirs(os.path.dirname(outputDirectoryPath2), exist_ok=True)

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    def region_selection(event, x, y, flags, param):

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed
        if event == cv.EVENT_LBUTTONDOWN:

            reactangle.append(x)
            reactangle.append(y)
            print(reactangle)

        elif event == cv.EVENT_MOUSEMOVE:

            leftx = reactangle[0]
            lefty = reactangle[1]

            cv.rectangle(img, (leftx, lefty), (x, y), (60, 20, 220), 2)
            cv.imshow("image", img)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            # record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished

            reactangle.append(x)
            reactangle.append(y)

            # draw a rectangle around the region of interest
            print(reactangle)
            cv.rectangle(img, (reactangle[0], reactangle[1]), (reactangle[2], reactangle[3]), (0, 255, 0), 2)
            cv.imshow("image", img)
            # reactangle_copy = reactangle.copy()

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            # denotes top left to downright motion
            acrossFlag = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            clone = img.copy()
            cv.namedWindow("image")
            cv.setMouseCallback("image", region_selection)

            # keep looping until the 'q' key is pressed
            while True:

                # display the image and wait for a input from keyboard
                cv.imshow("image", img)
                key = cv.waitKey(1) & 0xFF

                # if 'r' key is pressed, the window is reset and
                if key == ord("r"):
                    img = clone.copy()
                    reactangle = []
                    break

                # if the space key is pressed, break from the loop
                elif key == ord(" "):
                    print(reactangle)
                    break

                elif key == ord("n"):
                    reactangle = []
                    break

            if len(reactangle) == 4:

                # cropping the part of the image and showing that
                # copy to avoid reference issues

                if reactangle[1] < reactangle[3] and reactangle[0] < reactangle[2]:
                    # for left to right across
                    crop_img = clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :].copy()
                else:
                    # for right to left across
                    crop_img = clone[reactangle[1]:reactangle[3], reactangle[2]:reactangle[0], :].copy()
                    acrossFlag = 1

                # crop_img = clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :].copy()
                crop_img_rsz = cv.resize(crop_img, (512, 512))
                cv.imshow("crop_img", crop_img)
                cv.imshow("crop resized img", crop_img_rsz)

                # replacing the cropped part with white pixels
                if acrossFlag:
                    clone[reactangle[1]:reactangle[3], reactangle[2]:reactangle[0], :] = 255
                    reactangles = [min(reactangle[0], reactangle[2]), min(reactangle[1], reactangle[3]),
                                   max(reactangle[0], reactangle[2]), max(reactangle[1], reactangle[3])]
                    row = np.array([np.append(reactangles, file)])
                else:
                    clone[reactangle[1]:reactangle[3], reactangle[0]:reactangle[2], :] = 255
                    row = np.array([np.append(reactangle, file)])

                # saving the cropped image in output directory
                outFileName = outputDirectoryPath1 + prefix + file
                print("[ CROP: " + outFileName + " ]", end="\n\n\n")
                cv.imwrite(outFileName, crop_img)

                # saving the replaced image in the second ouput directory
                outFileName = outputDirectoryPath2 + prefix + file
                print("[ REPLACE: " + outFileName + " ]", end="\n\n\n")
                cv.imwrite(outFileName, clone)

                # saving reactangle values as rows in output csv file
                # row = np.array([np.append(reactangle, file)])

                np.savetxt(f, row, delimiter=',', fmt='%s')

                # resetting reactangle for next iteration
                reactangle = []

                cv.waitKey(0)

                # close all open windows
            cv.destroyAllWindows()
            cv.waitKey(1)

            # np.savetxt(f, row, delimiter=",")

            # index += 1

            # print("[ " + str(index) + ". " + fullFilePath + "]")

    f.close()


def get_GLCM_features_HSI(inputDirectoryPath, label, angles=[0, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180],
                           distances=[5], outputDirectoryPath="./", csvFileName="glcm_output.csv", colorFlag=0,
                           ksize=16, stepSize=1):
    """
    Extract GLCM features from images in HSI color space
    saves extracted features in provided CSV file.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input Directory where images are stored.

    label: Integer
        The corresponding labels of the images.

    angles: List of floating point angles -> [0, 45 * np.pi/180, 90 * np.pi/180, 135 * np.pi/180]
        The angles from which GLCM will be calculated.

    distances: list of integers -> [5]
        The distance between pixels that will be considered for GLCM.

    outputDirectoryPath: String -> "./"
        Path to the output directory where the feature CSV file will be saved.

    csvFileName: String -> "glcm_output.csv"
        The filename of the CSV file where the features will be saved.

    colorFlag: Integer -> 0
        Flag corresponding to the respective color space.

    ksize: Integer -> 16
        The set size for kernel height and width. As square shaped kernels are used, this values is both the kernel height and kernel width.

    stepSize: Integer -> 1
        The amount of pixels the kernel moves after classfying the contents of one position.

    ----------
    """

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    outputCSVFile = outputDirectoryPath + csvFileName

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    counter = 0

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")
        # distances = [1]
        angles = [0, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180]
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

        for file in fileNames:

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            clone = img.copy()
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                slice_h, slice_s, slice_v = cv.split(window)

                # if complete white window, don't register it
                if sum(slice_v.ravel()) >= 65000 and sum(slice_v.ravel()) <= 65280:
                    continue

                glcm_h = greycomatrix(slice_h, distances, angles, 256, symmetric=True, normed=True)
                glcm_s = greycomatrix(slice_s, distances, angles, 256, symmetric=True, normed=True)
                glcm_v = greycomatrix(slice_v, distances, angles, 256, symmetric=True, normed=True)

                for prop in props:
                    result_h = greycoprops(glcm_h, prop)
                    result_s = greycoprops(glcm_s, prop)
                    result_v = greycoprops(glcm_v, prop)

                    results.append([result_h, result_s, result_v])

                    # print(result_b)
                    # print(result_g)
                    # print(result_r)

                row = np.array([np.append(results, label)])

                # saving reactangle values as rows in output csv file
                np.savetxt(f, row, delimiter=',')

                clone = img.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)
                cv.waitKey(1)

                counter += 1

                # time.sleep(.0025)

    f.close()

    print("[ " + "Total " + str(counter) + " rows added ]")

    cv.destroyAllWindows()
    cv.waitKey(1)


def get_GLCM_features_RGB(inputDirectoryPath, label, angles=[0, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180],
                           distances=[5], outputDirectoryPath="./", csvFileName="glcm_output.csv", colorFlag=0,
                           ksize=16, stepSize=1):
    """
    Extract GLCM features from images in RGB color space
    saves extracted features in provided CSV file.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input Directory where images are stored.

    label: Integer
        The corresponding labels of the images.

    angles: List of floating point angles -> [0, 45 * np.pi/180, 90 * np.pi/180, 135 * np.pi/180]
        The angles from which GLCM will be calculated.

    distances: list of integers -> [5]
        The distance between pixels that will be considered for GLCM.

    outputDirectoryPath: String -> "./"
        Path to the output directory where the feature CSV file will be saved.

    csvFileName: String -> "glcm_output.csv"
        The filename of the CSV file where the features will be saved.

    colorFlag: Integer -> 0
        Flag corresponding to the respective color space.

    ksize: Integer -> 16
        The set size for kernel height and width. As square shaped kernels are used, this values is both the kernel height and kernel width.

    stepSize: Integer -> 1
        The amount of pixels the kernel moves after classfying the contents of one position.

    ----------
    """

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    outputCSVFile = outputDirectoryPath + csvFileName

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    counter = 0

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            clone = img.copy()
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                # distances = [1]
                # angles = [0, 45 * np.pi/180, 90 * np.pi/180, 135 * np.pi/180]
                props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

                slice_b, slice_g, slice_r = cv.split(window)

                glcm_b = greycomatrix(slice_b, distances, angles, 256, symmetric=True, normed=True)
                glcm_g = greycomatrix(slice_g, distances, angles, 256, symmetric=True, normed=True)
                glcm_r = greycomatrix(slice_r, distances, angles, 256, symmetric=True, normed=True)

                for prop in props:
                    result_b = greycoprops(glcm_b, prop)
                    result_g = greycoprops(glcm_g, prop)
                    result_r = greycoprops(glcm_r, prop)

                    results.append([result_b, result_g, result_r])

                    # print(result_b)
                    # print(result_g)
                    # print(result_r)

                row = np.array([np.append(results, label)])

                # saving reactangle values as rows in output csv file
                np.savetxt(f, row, delimiter=',')

                clone = img.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)
                cv.waitKey(1)

                counter += 1

                # time.sleep(.0025)

    f.close()

    print("[ " + "Total " + str(counter) + " rows added ]")

    cv.destroyAllWindows()
    cv.waitKey(1)


def get_CHPOOL_features_RGB(inputDirectoryPath, label, outputDirectoryPath="./",
                                         csvFileName="color_hist_shape_output.csv", colorFlag=0, ksize=16, stepSize=1):
    """
    Extract Color Histogram and MinPool and MaxPool features from images in RGB color space.
    Save extracted features in provided CSV file.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input images where the training images are stored.

    label: Integer
        Label corresponding to the extracted features.

    outputDirectoryPath: String -> "./"
        Path to the folder where the CSV file containing extracted features will be saved.

    csvFileName: String -> "color_hist_shape_output.csv"
        File name of the CSV file where the extracted features will be saved.

    colorFlag: Integer -> 0
        Flag corresponding to the color space used.

    ksize: Integer -> 16
        The set size for kernel height and width. As square shaped kernels are used, this values is both the kernel height and kernel width.

    stepSize: Integer -> 1
        The amount of pixels the kernel moves after classfying the contents of one position.
    ----------
    """

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    outputCSVFile = outputDirectoryPath + csvFileName

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    counter = 0

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            clone = img.copy()
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                # distances = [1]
                # angles = [0, 45 * np.pi/180, 90 * np.pi/180, 135 * np.pi/180]
                # props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

                chans = cv.split(window)

                hist_b = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
                hist_g = cv.calcHist([chans[1]], [0], None, [256], [0, 256])
                hist_r = cv.calcHist([chans[2]], [0], None, [256], [0, 256])

                minpool = block_reduce(window, (3, 3, 1), func=np.min)
                maxpool = block_reduce(window, (3, 3, 1), func=np.max)
                # avgpool = block_reduce(window, (3,3,1), func=np.mean)

                results.append([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
                results1 = np.append(results, minpool.flatten())
                results2 = np.append(results1, maxpool.flatten())
                row = np.array([np.append(results2, label)])

                # row = np.reshape(row, (row.shape[1], row.shape[0]))
                # print(row)
                # print("done")

                # saving reactangle values as rows in output csv file
                np.savetxt(f, [row.flatten()], delimiter=',', fmt="%s")

                clone = img.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)
                cv.waitKey(1)

                counter += 1

                # time.sleep(.0025)

    f.close()

    print("[ " + "Total " + str(counter) + " rows added ]")

    cv.destroyAllWindows()
    cv.waitKey(1)


def get_CHPOOL_features_YCrCb(inputDirectoryPath, label, outputDirectoryPath="./",
                                           csvFileName="color_hist_shape_output.csv", colorFlag=0, ksize=16,
                                           stepSize=1):
    """
    Extract Color Histogram and MinPool and MaxPool features from images in YCbCr color space.
    Save extracted features in provided CSV file.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input images where the training images are stored.

    label: Integer
        Label corresponding to the extracted features.

    outputDirectoryPath: String -> "./"
        Path to the folder where the CSV file containing extracted features will be saved.

    csvFileName: String -> "color_hist_shape_output.csv"
        File name of the CSV file where the extracted features will be saved.

    colorFlag: Integer -> 0
        Flag corresponding to the color space used.

    ksize: Integer -> 16
        The set size for kernel height and width. As square shaped kernels are used, this values is both the kernel height and kernel width.

    stepSize: Integer -> 1
        The amount of pixels the kernel moves after classfying the contents of one position.
    ----------
    """

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    outputCSVFile = outputDirectoryPath + csvFileName

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    counter = 0

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

            clone = img.copy()
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                # distances = [1]
                # angles = [0, 45 * np.pi/180, 90 * np.pi/180, 135 * np.pi/180]
                # props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

                chans = cv.split(window)

                hist_y = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
                hist_cr = cv.calcHist([chans[1]], [0], None, [256], [0, 256])
                hist_cb = cv.calcHist([chans[2]], [0], None, [256], [0, 256])

                minpool = block_reduce(window, (3, 3, 1), func=np.min)
                maxpool = block_reduce(window, (3, 3, 1), func=np.max)
                # avgpool = block_reduce(window, (3,3,1), func=np.mean)

                results.append([hist_y.flatten(), hist_cr.flatten(), hist_cb.flatten()])
                results1 = np.append(results, minpool.flatten())
                results2 = np.append(results1, maxpool.flatten())
                row = np.array([np.append(results2, label)])

                # row = np.reshape(row, (row.shape[1], row.shape[0]))
                # print(row)
                # print("done")

                # saving reactangle values as rows in output csv file
                np.savetxt(f, [row.flatten()], delimiter=',', fmt="%s")

                clone = img.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)
                cv.waitKey(1)

                counter += 1

                # time.sleep(.0025)

    f.close()

    print("[ " + "Total " + str(counter) + " rows added ]")

    cv.destroyAllWindows()
    cv.waitKey(1)


def get_CHPOOL_features_LAB(inputDirectoryPath, label, outputDirectoryPath="./",
                                         csvFileName="color_hist_shape_output.csv", colorFlag=0, ksize=16, stepSize=1):
    """
    Extract Color Histogram and MinPool and MaxPool features from images in LAB color space.
    Save extracted features in provided CSV file.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input images where the training images are stored.

    label: Integer
        Label corresponding to the extracted features.

    outputDirectoryPath: String -> "./"
        Path to the folder where the CSV file containing extracted features will be saved.

    csvFileName: String -> "color_hist_shape_output.csv"
        File name of the CSV file where the extracted features will be saved.

    colorFlag: Integer -> 0
        Flag corresponding to the color space used.

    ksize: Integer -> 16
        The set size for kernel height and width. As square shaped kernels are used, this values is both the kernel height and kernel width.

    stepSize: Integer -> 1
        The amount of pixels the kernel moves after classfying the contents of one position.
    ----------
    """

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    outputCSVFile = outputDirectoryPath + csvFileName

    print("[ OUTPUTFILE: " + outputCSVFile + " ]", end="\n\n\n")

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    counter = 0

    with open(outputCSVFile, 'ab') as f:

        # np.savetxt(f, columnNames, delimiter=",")

        for file in fileNames:

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            img = cv.cvtColor(img, cv.COLOR_BGR2LAB)

            clone = img.copy()
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                # distances = [1]
                # angles = [0, 45 * np.pi/180, 90 * np.pi/180, 135 * np.pi/180]
                # props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

                chans = cv.split(window)

                hist_l = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
                hist_a = cv.calcHist([chans[1]], [0], None, [256], [0, 256])
                hist_b = cv.calcHist([chans[2]], [0], None, [256], [0, 256])

                minpool = block_reduce(window, (3, 3, 1), func=np.min)
                maxpool = block_reduce(window, (3, 3, 1), func=np.max)
                # avgpool = block_reduce(window, (3,3,1), func=np.mean)

                results.append([hist_l.flatten(), hist_a.flatten(), hist_b.flatten()])
                results1 = np.append(results, minpool.flatten())
                results2 = np.append(results1, maxpool.flatten())
                row = np.array([np.append(results2, label)])

                # row = np.reshape(row, (row.shape[1], row.shape[0]))
                # print(row)
                # print("done")

                # saving reactangle values as rows in output csv file
                np.savetxt(f, [row.flatten()], delimiter=',', fmt="%s")

                clone = img.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)
                cv.waitKey(1)

                counter += 1

                # time.sleep(.0025)

    f.close()

    print("[ " + "Total " + str(counter) + " rows added ]")

    cv.destroyAllWindows()
    cv.waitKey(1)


def train_test_image_split(inputDirectoryPath, outputDirectoryPath=None, splitSize=0.7, trainDirectoryName="train",
                        testDirectoryName="test", seed=62):
    """

    Perform random subsampling of provided images based on given percentage.

    Parameters
    ----------
    inputDirectoryPath: String
        File path to the folder where the image data are stored.

    outputDirectoryPath: String
        File path to the folder where the images will be stored after splitting.

    splitSize = Float -> 70%
        The percentage of the standard train test split. By Default, 70% images are used for training, 30% images are kept from testing.

    trainDirectoryName : String -> "train"
        The name of the directory where the training images will be stored.

    testDirectoryName : String -> "test"
        The name of the directory where the test images will be stored.

    seed : Integer -> 62
        Seed used for the random subsampling used lated for reproducability.

    ----------

    """

    random.seed(seed)

    if outputDirectoryPath == None:
        outputDirectoryPath = inputDirectoryPath

    trainDirectoryPath = outputDirectoryPath + trainDirectoryName + "/"

    testDirectoryPath = outputDirectoryPath + testDirectoryName + "/"

    print("[ " + "Train split Directory set to: " + trainDirectoryPath + " ]")
    print("[ " + "Test split Directory set to: " + testDirectoryPath + " ]", end="\n\n")

    os.makedirs(os.path.dirname(trainDirectoryPath), exist_ok=True)
    os.makedirs(os.path.dirname(testDirectoryPath), exist_ok=True)

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    allFilePaths = [inputDirectoryPath + fileName for fileName in fileNames]

    numOfFiles = len(allFilePaths)
    numOfTrain = round(splitSize * numOfFiles)

    # random.sample(range(numOfFiles), numOfTrain)
    trainFilePaths = random.sample(allFilePaths, numOfTrain)
    testFilePaths = []
    for filePath in allFilePaths:
        if filePath not in trainFilePaths:
            testFilePaths.append(filePath)

    print(allFilePaths, end="\n\n\n")
    print(trainFilePaths, end="\n\n\n")
    print(testFilePaths, end="\n\n\n")

    trainCount = 0
    testCount = 0

    for file in trainFilePaths:
        copy(file, trainDirectoryPath)
        trainCount += 1

    for file in testFilePaths:
        copy(file, testDirectoryPath)
        testCount += 1

    print("[" + "Train split: " + str(trainCount) + " images" + "]")
    print("[" + "Test split: " + str(testCount) + " images" + "]", end="\n\n\n")


def draw_probable_LPRegions_GLCM_RGB(inputDirectoryPath, modelPath, outputDirectoryPath="./",
                             csvFileName="region_bounding_boxes.csv", colorFlag=0, ksize=16, stepSize=1,
                             confidence=0.90):
    """

    Detect and draw bounding box around the license plate region in RGB and save the resultant image and the bounding box information in CSV.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input directory

    outputDirectoryPath: String -> "./"
        Path to the output directory where the detected images will be stored.

    modelPath : String
        Path to the trained ML model

    csvFileNAme : String -> "region_bounding_boxes.csv"
        Name of the output CSV file that will hold the information regarding the images and their bounding boxes.

    colorFlag : Integer -> 0
        Flag determining the color space

    ksize : Integer -> 16
        Size of the kernel. As kernel is square shaped, this size is set as both the height and width of the kernel.

    stepSize : Integer -> 1
        Step size for the kernel. This is the amount the kernel moves after classifying one window.

    confidence : Float -> 90%
        The minimum threshold confidence set for identifying a region as LP region.

    ----------
    """

    model = joblib.load("rf_v2.joblib")
    ss = joblib.load("standard_scaler_v3.joblib")

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    with open(outputCSVFile, 'ab') as f:
        for file in fileNames:

            windows = []
            index = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            clone1 = img.copy()
            # clone1 = img[200:, 160:, :]
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                if (x < 160 or y < 200):
                    continue

                results = []

                distances = [1]
                angles = [0, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180]
                props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

                slice_b, slice_g, slice_r = cv.split(window)

                glcm_b = greycomatrix(slice_b, distances, angles, 256, symmetric=True, normed=True)
                glcm_g = greycomatrix(slice_g, distances, angles, 256, symmetric=True, normed=True)
                glcm_r = greycomatrix(slice_r, distances, angles, 256, symmetric=True, normed=True)

                for prop in props:
                    result_b = greycoprops(glcm_b, prop)
                    result_g = greycoprops(glcm_g, prop)
                    result_r = greycoprops(glcm_r, prop)

                    results.append([result_b, result_g, result_r])

                    # print(result_b)
                    # print(result_g)
                    # print(result_r)

                row = [np.ravel(results)]
                # print(row)
                row_scaled = ss.transform(row)

                # predict the class label with the model
                predictedLabel = model.predict(row_scaled)
                predictProba = model.predict_proba(row_scaled)

                # print(predictProba)

                # print(predictedLabel)

                if predictedLabel == 1 and predictProba[0][1] >= confidence:
                    print(predictProba)

                    downRightX = x + kW
                    downRightY = y + kH

                    if len(windows) == 0:
                        area = kW * kH
                        windows.append([x, y, downRightX, downRightY, area, index])
                        cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)
                        index += 1
                        continue

                    for window in windows:

                        if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("ahere\n")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        elif x + kH >= window[0] and x + kH <= window[2] and y >= window[1] and y <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere\n")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        elif x >= window[0] and x <= window[2] and y + kW >= window[1] and y + kW <= window[3]:

                            window[2] = downRightX
                            window[3] = downRightY
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        else:
                            windows.append([x, y, downRightX, downRightY, kW * kH, index])
                            index += 1

                    cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)

                # saving reactangle values as rows in output csv file
                # np.savetxt(f, row, delimiter = ',')

                clone = clone1.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)

                # cv.imshow("Window", img)
                cv.waitKey(1)

                # time.sleep(.0025)

            """
            for window in windows:



                if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                    window[0] = min(x, window[0])
                    window[1] = min(y, window[1])
                    window[2] = max(window[2], downRightX)
                    window[3] = max(window[3], downRightY)
                    window[4] = (window[2]-window[0])*(window[3]-window[1])
                    print("ahere\n")
                    print(window, end="\n")
                    #print(windows, end="\n")
                    windows[window[5]] = window

                elif x+kH >= window[0] and x+kH <= window[2] and y >= window[1] and y <= window[3]:
                    window[0] = min(x, window[0])
                    window[1] = min(y, window[1])
                    window[2] = max(window[2], downRightX)
                    window[3] = max(window[3], downRightY)
                    window[4] = (window[2]-window[0])*(window[3]-window[1])
                    print("bhere\n")
                    print(window, end="\n")
                    #print(windows, end="\n")
                    windows[window[5]] = window


                elif x >= window[0] and x <= window[2] and y+kW >= window[1] and y+kW <= window[3]:

                    window[2] = downRightX
                    window[3] = downRightY
                    window[4] = (window[2]-window[0])*(window[3]-window[1])
                    print("chere")
                    print(window, end="\n")
                    #print(windows, end="\n")
                    windows[window[5]] = window

                else:
                    windows.append([x, y, downRightX, downRightY, kW * kH, index])
                    index+=1
            """

            # print(windows)
            windows.sort(reverse=True, key=lambda x: x[4])
            # cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)
            if len(windows) != 0:
                cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0, 255, 0), 2)

                row = np.array([np.append([windows[0][0], windows[0][1], windows[0][2], windows[0][3]], file)])
                np.savetxt(f, row, delimiter=',', fmt='%s')
                print("saved text")
                print(row)

            # save the image
            # saving the cropped image in output directory
            outFileName = outputDirectoryPath + "lpd_" + file
            print("[ LPD: " + outFileName + " ]", end="\n\n\n")
            cv.imwrite(outFileName, clone1)

    f.close()

    cv.destroyAllWindows()
    cv.waitKey(1)


def draw_probable_LPRegions_GLCM_HSI(inputDirectoryPath, modelPath, outputDirectoryPath="./",
                             csvFileName="region_bounding_boxes.csv", colorFlag=0, ksize=16, stepSize=1,
                             confidence=0.90):
    """

    Detect and draw bounding box around the license plate region in HSI color space
    save the resultant image and the bounding box information in CSV.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input directory

    outputDirectoryPath: String -> "./"
        Path to the output directory where the detected images will be stored.

    modelPath : String
        Path to the trained ML model

    csvFileNAme : String -> "region_bounding_boxes.csv"
        Name of the output CSV file that will hold the information regarding the images and their bounding boxes.

    colorFlag : Integer -> 0
        Flag determining the color space

    ksize : Integer -> 16
        Size of the kernel. As kernel is square shaped, this size is set as both the height and width of the kernel.

    stepSize : Integer -> 1
        Step size for the kernel. This is the amount the kernel moves after classifying one window.

    confidence : Float -> 90%
        The minimum threshold confidence set for identifying a region as LP region.

    ----------
    """

    model = joblib.load("rf_HSI_v1.joblib")
    ss = joblib.load("ss_HSI_v1.joblib")

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    with open(outputCSVFile, 'ab') as f:

        distances = [1]
        angles = [0, 45 * np.pi / 180, 90 * np.pi / 180, 135 * np.pi / 180]
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

        for file in fileNames:

            windows = []
            index = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            clone1 = img.copy()
            # clone1 = img[200:, 160:, :]
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                slice_h, slice_s, slice_v = cv.split(window)

                if sum(slice_v.ravel()) >= 65000 and sum(slice_v.ravel()) <= 65280:
                    continue

                glcm_h = greycomatrix(slice_h, distances, angles, 256, symmetric=True, normed=True)
                glcm_s = greycomatrix(slice_s, distances, angles, 256, symmetric=True, normed=True)
                glcm_v = greycomatrix(slice_v, distances, angles, 256, symmetric=True, normed=True)

                for prop in props:
                    result_h = greycoprops(glcm_h, prop)
                    result_s = greycoprops(glcm_s, prop)
                    result_v = greycoprops(glcm_v, prop)

                    results.append([result_h, result_s, result_v])

                    # print(result_b)
                    # print(result_g)
                    # print(result_r)

                row = [np.ravel(results)]
                # print(row)
                row_scaled = ss.transform(row)

                # predict the class label with the model
                predictedLabel = model.predict(row_scaled)
                predictProba = model.predict_proba(row_scaled)

                # print(predictProba)

                # print(predictedLabel)

                if predictedLabel == 1 and predictProba[0][1] >= confidence:
                    print(predictProba)

                    downRightX = x + kW
                    downRightY = y + kH

                    if len(windows) == 0:
                        area = kW * kH
                        windows.append([x, y, downRightX, downRightY, area, index])
                        cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)
                        index += 1
                    else:
                        for window in windows:

                            if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("ahere\n")
                                print(window, end="\n")
                                print(len(windows))
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x + kH >= window[0] and x + kH <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere\n")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x >= window[0] and x <= window[2] and y + kW >= window[1] and y + kW <= window[3]:

                                window[2] = downRightX
                                window[3] = downRightY
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            else:
                                windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                index += 1

                    cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)

                # saving reactangle values as rows in output csv file
                # np.savetxt(f, row, delimiter = ',')

                clone = clone1.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)

                # cv.imshow("Window", img)
                cv.waitKey(1)

                # time.sleep(.0025)

            """
            for window in windows:



                if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                    window[0] = min(x, window[0])
                    window[1] = min(y, window[1])
                    window[2] = max(window[2], downRightX)
                    window[3] = max(window[3], downRightY)
                    window[4] = (window[2]-window[0])*(window[3]-window[1])
                    print("ahere\n")
                    print(window, end="\n")
                    #print(windows, end="\n")
                    windows[window[5]] = window

                elif x+kH >= window[0] and x+kH <= window[2] and y >= window[1] and y <= window[3]:
                    window[0] = min(x, window[0])
                    window[1] = min(y, window[1])
                    window[2] = max(window[2], downRightX)
                    window[3] = max(window[3], downRightY)
                    window[4] = (window[2]-window[0])*(window[3]-window[1])
                    print("bhere\n")
                    print(window, end="\n")
                    #print(windows, end="\n")
                    windows[window[5]] = window


                elif x >= window[0] and x <= window[2] and y+kW >= window[1] and y+kW <= window[3]:

                    window[2] = downRightX
                    window[3] = downRightY
                    window[4] = (window[2]-window[0])*(window[3]-window[1])
                    print("chere")
                    print(window, end="\n")
                    #print(windows, end="\n")
                    windows[window[5]] = window

                else:
                    windows.append([x, y, downRightX, downRightY, kW * kH, index])
                    index+=1
            """

            # print(windows)
            windows.sort(reverse=True, key=lambda x: x[4])
            # cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)
            if len(windows) != 0:
                cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0, 255, 0), 2)

                # changed in the HSI version
                # cv.rectangle(img, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)

                row = np.array([np.append([windows[0][0], windows[0][1], windows[0][2], windows[0][3]], file)])
                np.savetxt(f, row, delimiter=',', fmt='%s')
                print("saved text")
                print(row)

            # save the image
            # saving the cropped image in output directory
            outFileName = outputDirectoryPath + "lpd_" + file
            print("[ LPD: " + outFileName + " ]", end="\n\n\n")

            # changed in the HSI version
            cv.imwrite(outFileName, clone1)
            # cv.imwrite(outFileName, img)

    f.close()

    cv.destroyAllWindows()
    cv.waitKey(1)


def draw_probable_LPRegions_CHPOOL_RGB(inputDirectoryPath, modelPath, outputDirectoryPath="./",
                                                       csvFileName="region_bounding_boxes.csv", colorFlag=0, ksize=16,
                                                       stepSize=1, confidence=0.90):
    """

    Detect and draw bounding box around the license plate region based on Color Histogram and Pooling features in RGB color space.
    Save the resultant image with localized license plate and the bounding box information in CSV.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input directory

    outputDirectoryPath: String -> "./"
        Path to the output directory where the detected images will be stored.

    modelPath : String
        Path to the trained ML model

    csvFileNAme : String -> "region_bounding_boxes.csv"
        Name of the output CSV file that will hold the information regarding the images and their bounding boxes.

    colorFlag : Integer -> 0
        Flag determining the color space

    ksize : Integer -> 16
        Size of the kernel. As kernel is square shaped, this size is set as both the height and width of the kernel.

    stepSize : Integer -> 1
        Step size for the kernel. This is the amount the kernel moves after classifying one window.

    confidence : Float -> 90%
        The minimum threshold confidence set for identifying a region as LP region.

    ----------
    """

    model = joblib.load("./rf_NEW_CHPOOL_RGB_v1.joblib")
    ss = joblib.load("./ss_NEW_CHPOOL_RGB_v1.joblib")

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    with open(outputCSVFile, 'ab') as f:

        for file in fileNames:

            windows = []
            index = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            prevSum = 0

            clone1 = img.copy()
            # clone1 = img[200:, 160:, :]
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                chans = cv.split(window)

                hist_b = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
                hist_g = cv.calcHist([chans[1]], [0], None, [256], [0, 256])
                hist_r = cv.calcHist([chans[2]], [0], None, [256], [0, 256])

                minpool = block_reduce(window, (3, 3, 1), func=np.min)
                maxpool = block_reduce(window, (3, 3, 1), func=np.max)
                # avgpool = block_reduce(window, (3,3,1), func=np.mean)

                results.append([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
                results1 = np.append(results, minpool.flatten())
                results2 = np.append(results1, maxpool.flatten())

                row = [results2.flatten()]

                # print(row)
                row_scaled = ss.transform(row)

                # predict the class label with the model
                predictedLabel = model.predict(row_scaled)
                predictProba = model.predict_proba(row_scaled)

                # print(predictProba)

                # print(predictedLabel)

                if predictedLabel == 1 and predictProba[0][1] >= confidence:
                    print(predictProba)

                    downRightX = x + kW
                    downRightY = y + kH

                    if len(windows) == 0:
                        area = kW * kH
                        windows.append([x, y, downRightX, downRightY, area, index])

                        prevSum = np.sum([x, y, downRightX, downRightY, area])

                        cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)
                        print("here")
                        index += 1

                    else:
                        for window in windows:

                            if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("ahere\n")
                                print(window, end="\n")
                                # print(windows, end="\n\n\n")
                                print(len(windows))
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x + kH >= window[0] and x + kH <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere\n")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x >= window[0] and x <= window[2] and y + kW >= window[1] and y + kW <= window[3]:

                                window[2] = downRightX
                                window[3] = downRightY
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            else:
                                print("appending")
                                print(windows, end="\n\n\n")
                                nowSum = np.sum([x, y, downRightX, downRightY, kW * kH])
                                if nowSum != prevSum:
                                    windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                    prevSum = nowSum
                                    index += 1

                    cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)

                # saving reactangle values as rows in output csv file
                # np.savetxt(f, row, delimiter = ',')

                clone = clone1.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)

                # cv.imshow("Window", img)
                cv.waitKey(1)

                # time.sleep(.0025)

            for _ in range(1):

                for win in windows:

                    x, y, downRightX, downRightY = win[0], win[1], win[2], win[3]
                    topRightX, topRightY = max(x, downRightX), y
                    downLeftX, downLeftY = x, max(y, downRightY)

                    for window in windows:

                        # checking top left
                        if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("ahere\n")
                            print(window, end="\n")
                            # print(windows, end="\n\n\n")
                            print(len(windows))
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        # checking top right
                        elif topRightX >= window[0] and topRightX <= window[2] and topRightY >= window[
                            1] and topRightY <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere\n")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        # downLeft
                        elif downLeftX >= window[0] and downLeftX <= window[2] and downLeftY >= window[
                            1] and downLeftY <= window[3]:

                            window[2] = downRightX
                            window[3] = downRightY
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        """
                        else:
                            print("appending")
                            print(windows, end="\n\n\n")
                            nowSum = np.sum([x, y, downRightX, downRightY, kW * kH])
                            if nowSum != prevSum:
                                windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                prevSum = nowSum
                                index+=1
                        """

            tempWindows = []
            # print(windows)
            for window in windows:
                tempWindows.append([window[0], window[1], window[2], window[3], window[4]])

            windows = tempWindows.copy()

            uniqueWindows = []
            for window in windows:
                if window not in uniqueWindows:
                    uniqueWindows.append(window)

            windows = uniqueWindows.copy()
            windows.sort(reverse=True, key=lambda x: x[4])
            print("Printing 5 Possible Candidates")
            print(windows[:5])
            # cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)
            if len(windows) != 0:
                cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0, 255, 0), 2)

                # changed in the HSI version
                # cv.rectangle(img, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)

                row = np.array([np.append([windows[0][0], windows[0][1], windows[0][2], windows[0][3]], file)])
                np.savetxt(f, row, delimiter=',', fmt='%s')
                print("saved text")
                print(row)

            # save the image
            # saving the cropped image in output directory
            outFileName = outputDirectoryPath + "lpd_" + file
            print("[ LPD: " + outFileName + " ]", end="\n\n\n")

            # changed in the HSI version
            cv.imwrite(outFileName, clone1)
            # cv.imwrite(outFileName, img)

    f.close()

    cv.destroyAllWindows()
    cv.waitKey(1)


def draw_probable_LPRegions_CHPOOL_LAB(inputDirectoryPath, modelPath, outputDirectoryPath="./",
                                                       csvFileName="region_bounding_boxes.csv", colorFlag=0, ksize=16,
                                                       stepSize=1, confidence=0.90):
    """

    Detect and draw bounding box around the license plate region based on Color Histogram and Pooling features in LAB color space.
    Save the resultant image with localized license plate and the bounding box information in CSV.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input directory

    outputDirectoryPath: String -> "./"
        Path to the output directory where the detected images will be stored.

    modelPath : String
        Path to the trained ML model

    csvFileNAme : String -> "region_bounding_boxes.csv"
        Name of the output CSV file that will hold the information regarding the images and their bounding boxes.

    colorFlag : Integer -> 0
        Flag determining the color space

    ksize : Integer -> 16
        Size of the kernel. As kernel is square shaped, this size is set as both the height and width of the kernel.

    stepSize : Integer -> 1
        Step size for the kernel. This is the amount the kernel moves after classifying one window.

    confidence : Float -> 90%
        The minimum threshold confidence set for identifying a region as LP region.

    ----------
    """

    model = joblib.load("./rf_NEW_CHPOOL_LAB_v1.joblib")
    ss = joblib.load("./ss_NEW_CHPOOL_LAB_v1.joblib")

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    with open(outputCSVFile, 'ab') as f:

        for file in fileNames:

            windows = []
            index = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            prevSum = 0

            clone1 = img.copy()

            img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            # clone1 = img[200:, 160:, :]
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                chans = cv.split(window)

                hist_l = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
                hist_a = cv.calcHist([chans[1]], [0], None, [256], [0, 256])
                hist_b = cv.calcHist([chans[2]], [0], None, [256], [0, 256])

                minpool = block_reduce(window, (3, 3, 1), func=np.min)
                maxpool = block_reduce(window, (3, 3, 1), func=np.max)
                # avgpool = block_reduce(window, (3,3,1), func=np.mean)

                results.append([hist_l.flatten(), hist_a.flatten(), hist_b.flatten()])
                results1 = np.append(results, minpool.flatten())
                results2 = np.append(results1, maxpool.flatten())

                row = [results2.flatten()]

                # print(row)
                row_scaled = ss.transform(row)

                # predict the class label with the model
                predictedLabel = model.predict(row_scaled)
                predictProba = model.predict_proba(row_scaled)

                # print(predictProba)

                # print(predictedLabel)

                if predictedLabel == 1 and predictProba[0][1] >= confidence:
                    print(predictProba)

                    downRightX = x + kW
                    downRightY = y + kH

                    if len(windows) == 0:
                        area = kW * kH
                        windows.append([x, y, downRightX, downRightY, area, index])

                        prevSum = np.sum([x, y, downRightX, downRightY, area])

                        cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)
                        print("here")
                        index += 1

                    else:
                        for window in windows:

                            if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("ahere\n")
                                print(window, end="\n")
                                # print(windows, end="\n\n\n")
                                print(len(windows))
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x + kH >= window[0] and x + kH <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere\n")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x >= window[0] and x <= window[2] and y + kW >= window[1] and y + kW <= window[3]:

                                window[2] = downRightX
                                window[3] = downRightY
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            else:
                                print("appending")
                                print(windows, end="\n\n\n")
                                nowSum = np.sum([x, y, downRightX, downRightY, kW * kH])
                                if nowSum != prevSum:
                                    windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                    prevSum = nowSum
                                    index += 1

                    cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)

                # saving reactangle values as rows in output csv file
                # np.savetxt(f, row, delimiter = ',')

                clone = clone1.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)

                # cv.imshow("Window", img)
                cv.waitKey(1)

                # time.sleep(.0025)

            for _ in range(1):

                for win in windows:

                    x, y, downRightX, downRightY = win[0], win[1], win[2], win[3]
                    topRightX, topRightY = max(x, downRightX), y
                    downLeftX, downLeftY = x, max(y, downRightY)

                    for window in windows:

                        # checking top left
                        if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("ahere\n")
                            print(window, end="\n")
                            # print(windows, end="\n\n\n")
                            print(len(windows))
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        # checking top right
                        elif topRightX >= window[0] and topRightX <= window[2] and topRightY >= window[
                            1] and topRightY <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere\n")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        # downLeft
                        elif downLeftX >= window[0] and downLeftX <= window[2] and downLeftY >= window[
                            1] and downLeftY <= window[3]:

                            window[2] = downRightX
                            window[3] = downRightY
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        """
                        else:
                            print("appending")
                            print(windows, end="\n\n\n")
                            nowSum = np.sum([x, y, downRightX, downRightY, kW * kH])
                            if nowSum != prevSum:
                                windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                prevSum = nowSum
                                index+=1
                        """

            tempWindows = []
            # print(windows)
            for window in windows:
                tempWindows.append([window[0], window[1], window[2], window[3], window[4]])

            windows = tempWindows.copy()

            uniqueWindows = []
            for window in windows:
                if window not in uniqueWindows:
                    uniqueWindows.append(window)

            windows = uniqueWindows.copy()
            windows.sort(reverse=True, key=lambda x: x[4])
            print("Printing 5 Possible Candidates")
            print(windows[:5])
            # cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)
            if len(windows) != 0:
                cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0, 255, 0), 2)

                # changed in the HSI version
                # cv.rectangle(img, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)

                row = np.array([np.append([windows[0][0], windows[0][1], windows[0][2], windows[0][3]], file)])
                np.savetxt(f, row, delimiter=',', fmt='%s')
                print("saved text")
                print(row)

            # save the image
            # saving the cropped image in output directory
            outFileName = outputDirectoryPath + "lpd_" + file
            print("[ LPD: " + outFileName + " ]", end="\n\n\n")

            # changed in the HSI version
            cv.imwrite(outFileName, clone1)
            # cv.imwrite(outFileName, img)

    f.close()

    cv.destroyAllWindows()
    cv.waitKey(1)


def draw_probable_LPRegions_CHPOOL_YCrCb(inputDirectoryPath, modelPath, outputDirectoryPath="./",
                                                         csvFileName="region_bounding_boxes.csv", colorFlag=0, ksize=16,
                                                         stepSize=1, confidence=0.90):
    """

    Detect and draw bounding box around the license plate region based on Color Histogram and Pooling features in YCrCb color space.
    Save the resultant image with localized license plate and the bounding box information in CSV.

    Parameters
    ----------
    inputDirectoryPath: String
        Path to the input directory

    outputDirectoryPath: String -> "./"
        Path to the output directory where the detected images will be stored.

    modelPath : String
        Path to the trained ML model

    csvFileNAme : String -> "region_bounding_boxes.csv"
        Name of the output CSV file that will hold the information regarding the images and their bounding boxes.

    colorFlag : Integer -> 0
        Flag determining the color space

    ksize : Integer -> 16
        Size of the kernel. As kernel is square shaped, this size is set as both the height and width of the kernel.

    stepSize : Integer -> 1
        Step size for the kernel. This is the amount the kernel moves after classifying one window.

    confidence : Float -> 90%
        The minimum threshold confidence set for identifying a region as LP region.

    ----------
    """

    model = joblib.load("./rf_NEW_CHPOOL_YCrCb_v1.joblib")
    ss = joblib.load("./ss_NEW_CHPOOL_YCrCb_v1.joblib")

    def sliding_window(image, stepSize, windowSize):
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    kW = ksize
    kH = ksize

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    with open(outputCSVFile, 'ab') as f:

        for file in fileNames:

            windows = []
            index = 0

            fullFilePath = inputDirectoryPath + file
            print("[ INPUT: " + fullFilePath + " ]")

            img = cv.imread(fullFilePath)

            prevSum = 0

            clone1 = img.copy()

            img = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

            # clone1 = img[200:, 160:, :]
            # cv.namedWindow("image")

            for (x, y, window) in sliding_window(img, stepSize=stepSize, windowSize=(kW, kH)):

                if window.shape[0] != kH or window.shape[1] != kW:
                    continue

                results = []

                chans = cv.split(window)

                hist_y = cv.calcHist([chans[0]], [0], None, [256], [0, 256])
                hist_cr = cv.calcHist([chans[1]], [0], None, [256], [0, 256])
                hist_cb = cv.calcHist([chans[2]], [0], None, [256], [0, 256])

                minpool = block_reduce(window, (3, 3, 1), func=np.min)
                maxpool = block_reduce(window, (3, 3, 1), func=np.max)
                # avgpool = block_reduce(window, (3,3,1), func=np.mean)

                results.append([hist_y.flatten(), hist_cr.flatten(), hist_cb.flatten()])
                results1 = np.append(results, minpool.flatten())
                results2 = np.append(results1, maxpool.flatten())

                row = [results2.flatten()]

                # print(row)
                row_scaled = ss.transform(row)

                # predict the class label with the model
                predictedLabel = model.predict(row_scaled)
                predictProba = model.predict_proba(row_scaled)

                # print(predictProba)

                # print(predictedLabel)

                if predictedLabel == 1 and predictProba[0][1] >= confidence:
                    print(predictProba)

                    downRightX = x + kW
                    downRightY = y + kH

                    if len(windows) == 0:
                        area = kW * kH
                        windows.append([x, y, downRightX, downRightY, area, index])

                        prevSum = np.sum([x, y, downRightX, downRightY, area])

                        cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)
                        print("here")
                        index += 1

                    else:
                        for window in windows:

                            if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("ahere\n")
                                print(window, end="\n")
                                # print(windows, end="\n\n\n")
                                print(len(windows))
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x + kH >= window[0] and x + kH <= window[2] and y >= window[1] and y <= window[3]:
                                window[0] = min(x, window[0])
                                window[1] = min(y, window[1])
                                window[2] = max(window[2], downRightX)
                                window[3] = max(window[3], downRightY)
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere\n")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            elif x >= window[0] and x <= window[2] and y + kW >= window[1] and y + kW <= window[3]:

                                window[2] = downRightX
                                window[3] = downRightY
                                window[4] = (window[2] - window[0]) * (window[3] - window[1])
                                print("bhere")
                                print(window, end="\n")
                                # print(windows, end="\n")
                                windows[window[5]] = window

                            else:
                                print("appending")
                                print(windows, end="\n\n\n")
                                nowSum = np.sum([x, y, downRightX, downRightY, kW * kH])
                                if nowSum != prevSum:
                                    windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                    prevSum = nowSum
                                    index += 1

                    cv.rectangle(clone1, (x, y), (x + kW, y + kH), (255, 0, 0), 2)

                # saving reactangle values as rows in output csv file
                # np.savetxt(f, row, delimiter = ',')

                clone = clone1.copy()
                cv.rectangle(clone, (x, y), (x + kW, y + kH), (0, 255, 0), 2)
                cv.imshow("Window", clone)

                # cv.imshow("Window", img)
                cv.waitKey(1)

                # time.sleep(.0025)

            for _ in range(1):

                for win in windows:

                    x, y, downRightX, downRightY = win[0], win[1], win[2], win[3]
                    topRightX, topRightY = max(x, downRightX), y
                    downLeftX, downLeftY = x, max(y, downRightY)

                    for window in windows:

                        # checking top left
                        if x >= window[0] and x <= window[2] and y >= window[1] and y <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("ahere\n")
                            print(window, end="\n")
                            # print(windows, end="\n\n\n")
                            print(len(windows))
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        # checking top right
                        elif topRightX >= window[0] and topRightX <= window[2] and topRightY >= window[
                            1] and topRightY <= window[3]:
                            window[0] = min(x, window[0])
                            window[1] = min(y, window[1])
                            window[2] = max(window[2], downRightX)
                            window[3] = max(window[3], downRightY)
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere\n")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        # downLeft
                        elif downLeftX >= window[0] and downLeftX <= window[2] and downLeftY >= window[
                            1] and downLeftY <= window[3]:

                            window[2] = downRightX
                            window[3] = downRightY
                            window[4] = (window[2] - window[0]) * (window[3] - window[1])
                            print("bhere")
                            print(window, end="\n")
                            # print(windows, end="\n")
                            windows[window[5]] = window

                        """
                        else:
                            print("appending")
                            print(windows, end="\n\n\n")
                            nowSum = np.sum([x, y, downRightX, downRightY, kW * kH])
                            if nowSum != prevSum:
                                windows.append([x, y, downRightX, downRightY, kW * kH, index])
                                prevSum = nowSum
                                index+=1
                        """

            tempWindows = []
            # print(windows)
            for window in windows:
                tempWindows.append([window[0], window[1], window[2], window[3], window[4]])

            windows = tempWindows.copy()

            uniqueWindows = []
            for window in windows:
                if window not in uniqueWindows:
                    uniqueWindows.append(window)

            windows = uniqueWindows.copy()
            windows.sort(reverse=True, key=lambda x: x[4])
            print("Printing 5 Possible Candidates")
            print(windows[:5])
            # cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)
            if len(windows) != 0:
                cv.rectangle(clone1, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0, 255, 0), 2)

                # changed in the HSI version
                # cv.rectangle(img, (windows[0][0], windows[0][1]), (windows[0][2], windows[0][3]), (0,255,0), 2)

                row = np.array([np.append([windows[0][0], windows[0][1], windows[0][2], windows[0][3]], file)])
                np.savetxt(f, row, delimiter=',', fmt='%s')
                print("saved text")
                print(row)

            # save the image
            # saving the cropped image in output directory
            outFileName = outputDirectoryPath + "lpd_" + file
            print("[ LPD: " + outFileName + " ]", end="\n\n\n")

            # changed in the HSI version
            cv.imwrite(outFileName, clone1)
            # cv.imwrite(outFileName, img)

    f.close()

    cv.destroyAllWindows()
    cv.waitKey(1)


def __getShapes__(inputDirectoryPath):
    """
    Calculate and draw a histogram of the shape information of the images.

    Parameters
    ----------
    inputDirectoryPath : String
        The path to the input directory
    ----------

    Usage
    -----
    -----

    """

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    shapeArray = []

    for file in fileNames:
        fullFilePath = inputDirectoryPath + file

        img = cv.imread(fullFilePath)

        h, w, _ = img.shape

        shapeArray.append((h, w))

    print("[ Min Height : " + str(min(np.array(shapeArray)[:, :1])) + " ]")
    print("[ Max Height : " + str(max(np.array(shapeArray)[:, :1])) + " ]")

    print("[ Min Width : " + str(min(np.array(shapeArray)[:, [1]])) + " ]")
    print("[ Max Width : " + str(max(np.array(shapeArray)[:, [1]])) + " ]")

    # print(np.array(shapeArray)[:,:1])

    sns.distplot(np.array(shapeArray)[:, :1], kde=False)

    return shapeArray


def batch_resize(inputDirectoryPath, outputDirectoryPath, prefix="resized", keepAspectRatio=False):
    """

    Resize all the image in input directory and save the resized images in output directory.

    Parameters
    ----------
    inputDirectoryPath : String
        Path to the input images directory.

    outputDirectoryPath : String
        Path to the directory where the resized images will be saved.

    prefix : String -> "resized"
        The prefix of the saved images.

    keepAspectRatio : Boolean
        Keep aspect ratio while resizing.

    ----------
    """

    inter = cv.INTER_TAB_SIZE

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)

    for file in fileNames:

        fullFilePath = inputDirectoryPath + file
        print("[ INPUT: " + fullFilePath + " ]")

        img = cv.imread(fullFilePath)

        h, w, _ = img.shape

        if keepAspectRatio:
            if h > w:
                img_resize = cv.resize(img, (480, 640), inter)

            elif w > h:
                img_resize = cv.resize(img, (640, 480), inter)

            else:
                img_resize = cv.resize(img, (640, 480), inter)
        else:
            img_resize = cv.resize(img, (640, 480), inter)

        outFileName = outputDirectoryPath + prefix + file
        print("[ Resized: " + outFileName + " ]", end="\n\n\n")
        cv.imwrite(outFileName, img_resize)


def compute_IoU(detectionResultFilePath, gtFilePath, outputDirectoryPath, csvFileName, imagePath,
               annotOutputDirectoryPath, iouThreshold=0.50):
    """
    Calculate IoU (Intersection of Union) and TP, TN, FP, FN given two CSV files:
        CSV file of the bounding box result from the outputs of test images.
        CSV file of the ground truth information for the test images.

    Parameters
    ----------
    detectionResultFilePath : String
        Path to the CSV files containing bounding box information of the test images. These are the outputs of the Detection and localization system.

    gtFilePath : String
        Path to the CSV files containing the ground truth information.

    outputDirectoryPath : String
        Path to the output files

    csvFileName : String
        Filename of the CSV file contining the extracted feature vectors.

    imagePath : String
        Path to the original test image path that containts the original test images..

    annotOutputDirectoryPath : String
        Path to the output directory where the annotated images with detected LP and IoU will be output.

    iouThreshold : String
        Threshold set for calculating Detection or Non-Detection results.

    ----------

    """

    os.makedirs(os.path.dirname(annotOutputDirectoryPath), exist_ok=True)

    font_scale = 3
    font = cv.FONT_HERSHEY_PLAIN

    # set the rectangle background to white
    rectangle_bgr = (255, 255, 0)

    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        Returns
        -------
        float
            in [0, 1]
        """

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    detectionResult = np.array(pd.read_csv(detectionResultFilePath))
    gtResult = np.array(pd.read_csv(gtFilePath))

    os.makedirs(os.path.dirname(outputDirectoryPath), exist_ok=True)
    outputCSVFile = outputDirectoryPath + csvFileName

    with open(outputCSVFile, 'ab') as f:
        for gt in gtResult:
            index = np.where((detectionResult[:, [4]] == gt))

            if len(index[0]) != 0:
                detectionIndex = index[0][0]
                det = detectionResult[detectionIndex]

                iou = get_iou(gt, det)

                # get image
                fullImagePath = imagePath + det[4]

                imgAnnot = cv.imread(fullImagePath)

                # draw green for gt, draw red for detection
                cv.rectangle(imgAnnot, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 2)
                cv.rectangle(imgAnnot, (det[0], det[1]), (det[2], det[3]), (0, 0, 255), 2)

                text = "".join(list(str(iou))[:7])
                # get the width and height of the text box
                (text_width, text_height) = cv.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
                # set the text start position
                text_offset_x = 30
                # text_offset_y = imgAnnot.shape[0] - 25
                text_offset_y = 40
                # make the coords of the box with a small padding of two pixels
                box_coords = (
                (text_offset_x, text_offset_y), (text_offset_x + text_width + 5, text_offset_y - text_height - 5))
                cv.rectangle(imgAnnot, box_coords[0], box_coords[1], rectangle_bgr, cv.FILLED)
                cv.putText(imgAnnot, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0),
                           thickness=1)

                # cv.putText(imgAnnot, str(round(iou)), (30,30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                # save annotated version of image
                outFileName = annotOutputDirectoryPath + "annot_" + det[4]
                print("[ ANNOTATION: " + outFileName + " ]", end="\n\n\n")
                cv.imwrite(outFileName, imgAnnot)

                if iou > iouThreshold:

                    label = [1, "TP"]

                    results = np.array([np.append(gt, det[0:4])])
                    results1 = np.array([np.append(results, iou)])
                    row = np.array([np.append(results1, label)])

                    print(row)

                    np.savetxt(f, row, delimiter=',', fmt='%s')

                else:
                    label = [0, "FN"]

                    results = np.array([np.append(gt, det[0:4])])
                    results1 = np.array([np.append(results, iou)])
                    row = np.array([np.append(results1, label)])

                    print(row)

                    np.savetxt(f, row, delimiter=',', fmt='%s')

def __copyDir__(copyFromDirectoryPath, copyToDirectoryPath):
    """
    Copy files from a given input directory to output Directory

    Parameters
    ----------
    copyFromDirectoryPath : String
        Path to the directory from where the images will be copied.

    copyToDirectoryPath : String
        Path to the directory where the images will be copied.

    ----------
    """

    os.makedirs(os.path.dirname(copyToDirectoryPath), exist_ok=True)

    fileNamess = os.listdir(copyFromDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))

    for file in fileNames:
        fullFilePath = copyFromDirectoryPath + file

        copy(fullFilePath, copyToDirectoryPath)


def view_images(inputDirectoryPath, height, width):
    """
    View images in the provided size from a given directory

    Parameters
    ----------
    inputDirectoryPath : String
        Path to the directory from where the images will be loaded.

    height : int
        Resized height of the viewed images.

    width:int
        Resized width of the viewed images.

    ----------
    """

    heightOneThird = int(height * (1 / 3))
    heightTwoThird = int(height * (2 / 3))
    widthOneThird = int(width * (1 / 3))
    widthTwoThird = int(width * (2 / 3))

    fileNamess = os.listdir(inputDirectoryPath)

    fileNames = list(filter(lambda x: x.endswith('.jpg') or x.endswith(".JPG"), fileNamess))
    fileNames.sort(key=len)

    for file in fileNames:
        fullFilePath = inputDirectoryPath + file

        img = cv.imread(fullFilePath)

        img = cv.line(img, (0, heightOneThird), (width, heightOneThird), (0, 0, 255), 2)
        img = cv.line(img, (0, heightTwoThird), (width, heightTwoThird), (0, 255, 0), 2)

        img = cv.line(img, (widthOneThird, 0), (widthOneThird, height), (0, 0, 255), 2)
        img = cv.line(img, (widthTwoThird, 0), (widthTwoThird, height), (0, 255, 0), 2)

        cv.imshow(file, img)
        cv.waitKey(0)
        cv.destroyWindow(file)

    cv.destroyAllWindows()
    cv.waitKey(1)

