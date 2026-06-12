# MVAA Task 1 Solution

This repository contains our solution pipeline for **Task 1: Mitral Valve Segmentation and Landmark Localization in CT scans** of the MVAA challenge.

The initial goal is to build a clean baseline pipeline for:

- loading Task 1 CT data,
- training a 3D segmentation model,
- validating segmentation performance,
- generating prediction masks,
- exporting submission files in the official baseline format.

## Project Focus

Task 1 uses cardiac CT volumes. The model predicts mitral valve segmentation masks from 3D CT images.

Current data mapping:

```text
data/t1_ct/train/images      -> labeled training CT images
data/t1_ct/train/labels      -> labeled training segmentation masks
data/t1_ct/unlabeled/images  -> unlabeled training CT images
data/t1_ct/val/images        -> validation CT images



