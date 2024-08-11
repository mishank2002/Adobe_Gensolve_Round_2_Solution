# Adobe Gensolve Round 2 Solution

## Overview
This repository provides a solution for the Adobe Gensolve Round 2 challenge, which involves processing curves, identifying and regularizing shapes, exploring symmetries, completing incomplete curves, and comparing results with provided solutions.

## Step-by-Step Guide

### 1. Reading the Input Data
- **Objective:** Load and read the provided CSV files that contain raw data points representing polylines (curves).
- **Method:** Utilize Python libraries to parse and access the data points.

### 2. Regularizing Curves
- **Objective:** Identify and straighten common geometric shapes within the curves, including straight lines, circles, ellipses, rectangles, polygons, and star shapes.
- **Method:** Apply mathematical techniques to detect and regularize these shapes based on the properties of the data points.

### 3. Exploring Symmetry
- **Objective:** Investigate reflection symmetries in closed shapes and determine lines of symmetry.
- **Method:** Compare points on either side of potential symmetry axes to identify mirrored halves.

### 4. Completing Incomplete Curves
- **Objective:** Develop algorithms to fill in gaps within curves caused by occlusion.
- **Method:** Handle various levels of occlusion, from fully contained to partially contained and disconnected shapes. Use interpolation methods to complete the curves.

### 5. Comparing with Expected Results
- **Objective:** Validate the accuracy of your processed curves by comparing them with provided solution files.
- **Method:** Use the provided solution files (e.g., *_sol.csv and *_sol.svg) to assess the accuracy of your solutions by visualizing and comparing them to the references.

## Deep Dive

### 1. Reading CSV Files
- **Details:** Load the CSV files to access the data points for processing.

### 2. Identifying Shapes
- **Details:** Implement algorithms to detect and regularize shapes. For example, use the least squares method to fit circles to the data points.

### 3. Exploring Symmetry
- **Details:** Identify lines of symmetry by comparing points on either side of potential symmetry axes.

### 4. Completing Curves
- **Details:** Use interpolation methods, such as spline interpolation, to fill gaps and smoothly connect points in incomplete curves.

## Tools and Libraries

- **NumPy:** For numerical operations and handling arrays.
- **Matplotlib:** For plotting and visualizing curves.
- **SciPy:** For advanced mathematical operations like curve fitting and interpolation.


