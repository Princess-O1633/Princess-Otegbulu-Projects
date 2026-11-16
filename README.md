# CSULA Microscopy Image Analysis

Python scripts, microscopy images, and analysis spreadsheets from my CSULA work with alpha-synuclein samples.

The repository preserves the development process: image-processing experiments, batch-analysis variants, calibration notes, and exported measurements. Start with the links below to explore the work.

## Explore the project

| Material | Where to find it |
| --- | --- |
| Guide to the scripts and their outputs | [Project map](docs/PROJECT_MAP.md) |
| Batch image-analysis scripts | [neat_scatterall.py](neat_scatterall.py), [PBSall.py](PBSall.py), and [scatterall.py](scatterall.py) |
| Microscopy image sets | [2500X Batch 1&2](2500X%20Batch%201%262/), [grouped image sets](GROUPS%20-%202500X%20Batch%201%262/), and [PBS images](PBS%202500X/) |
| Saved spreadsheets and CSV files | [EXCEL SHEETS](EXCEL%20SHEETS/) |
| Scale-bar calibration notes | [scale_correlation.txt](scale_correlation.txt) |
| GitHub basics for maintaining this project | [Using this repository](docs/USING_GITHUB.md) |

## What the analysis code does

The batch-analysis variants include:

- Image denoising, contrast enhancement, and local entropy calculations.
- Thresholding and connected-component detection.
- OCR-assisted identification of image text and scale-bar regions.
- Conversion of component pixel measurements to micrometers using calibration values.
- Component area and maximum straight-line distance measurements, plots, and spreadsheet exports.

The reported straight-line distance is the maximum separation between component pixels; it is not a traced contour length. Exact processing choices differ between scripts.

## Working with the code

This is a development archive rather than an installable software package. Several scripts use fixed input paths, a Windows Tesseract installation path, and interactive plot windows. See the [project map](docs/PROJECT_MAP.md#running-the-historical-scripts) before running a script.

The saved spreadsheets are historical files. Their presence does not establish which exact script version and settings generated them. The project map records the output filenames found in code without claiming that those runs have been reproduced.

## Repository history

The original files were stored on the [part-2 branch](https://github.com/Princess-O1633/Princess-Otegbulu-Projects/tree/part-2). They are now also available on the default `main` branch, with their contents and paths preserved. The original branch remains available.

The [learning](learning/) and [tutorial](tutorial/) folders contain practice material and exploratory examples; they are listed separately from the research-analysis scripts in the project map.
