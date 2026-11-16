# Project map

[Back to the project](../README.md)

This guide describes the files present in the repository and the settings found in the source code. It does not select an authoritative final analysis version or infer scientific conclusions from filenames.

## Research-analysis scripts

These scripts contain related versions of the microscopy workflow. Similar filenames do not mean that their processing parameters are identical.

| Script | Input configured in the source | Output filename configured in the source |
| --- | --- | --- |
| [neat_scatterall.py](../neat_scatterall.py) | Files directly inside `2500X Batch 1&2` | `2500X_area_length.xlsx` |
| [PBSall.py](../PBSall.py) | Files directly inside `PBS 2500X` | `PBS_DATA.xlsx` |
| [scatterall.py](../scatterall.py) | `GROUP -D23aNP cover slip 2500x Full coaxial` inside the grouped image directory | `GROUP -D23aNP cover slip 2500x Full coaxial.xlsx` |
| [final/improving.py](../final/improving.py) | Direct entries in the grouped image directory; the pattern does not recurse into its subfolders | `GROUP -D23aNP cover slip 2500x Full coaxial.xlsx` |
| [cantseeem.py](../cantseeem.py) | A Condition 2 NP image pattern under `2500X Batch 1&2`; the matching group is currently stored in the separate grouped image directory | `GROUP_Condition_2_NP_2500x_partial_coaxial_5825.xlsx` |
| [original.py](../original.py) | Image supplied with the `--file` argument; manual scale selection is supported | Interactive measurements and plots |

Output filenames above are code settings, not verified links to the saved historical spreadsheets. The Excel-writing scripts use relative filenames, so exports are written to the working directory.

### Additional development versions

The [scripts](../scripts/) directory includes preprocessing experiments and earlier analysis variants:

| Files | Role visible in the code |
| --- | --- |
| `VERSION1.py`, `VERSION 2.py`, `VERSION 4.py`, `VERSION 5.py`, `fixing.py` | Segmentation and measurement experiments |
| `autothresh.py`, `feat.py`, `texture.py` | Thresholding, image enhancement, and texture experiments |
| `scanner.py` | OCR and image-processing experiment |
| `batch.py`, `presented.py` | Batch processing configured for a `photos/Batches` path that is absent from this snapshot |
| `scatter.py` | Batch analysis configured to export `fibril_analysis_output.xlsx` |
| `pls.py` | PBS image analysis configured to export `entropy.xlsx` |
| `xlsv.py` | Single-image analysis with an absolute Windows input path, configured to export `batches.xlsx` |

Names such as `presented.py` and `final/improving.py` have been retained from the original workspace. They do not, by themselves, establish which script was used for a presentation or final result.

## Images and saved measurements

| Location | Contents |
| --- | --- |
| [2500X Batch 1&2](../2500X%20Batch%201%262/) | Microscopy image set |
| [GROUPS - 2500X Batch 1&2](../GROUPS%20-%202500X%20Batch%201%262/) | Images arranged in sample-named subfolders |
| [PBS 2500X](../PBS%202500X/) | PBS and coverslip image files |
| [EXCEL SHEETS](../EXCEL%20SHEETS/) | Historical Excel workbooks and CSV exports; `output.csv` is empty |
| [scale_correlation.txt](../scale_correlation.txt) | Scale-bar measurements and calibration notes |

Two additional microscopy images are stored at the repository root. Original image and sample names are retained; abbreviations and experimental conditions should be interpreted using the lab records.

## Practice and local workspace files

- [learning](../learning/) contains OpenCV practice, example media, and exploratory fibril-analysis files such as `connect.py` and `main.py`.
- [tutorial](../tutorial/) contains image-processing practice scripts.
- `Tesseract-OCR` contains Windows shortcuts, not the Tesseract program itself.
- Editor settings, Windows metadata, and a workspace file are retained from the original snapshot.

## Running the historical scripts

The code has been inspected to document its configuration. An end-to-end rerun has not been performed as part of this organization work.

1. Download the repository from **Code → Download ZIP**, then extract it.
2. Open a terminal in the extracted repository's top-level folder. Many scripts resolve image paths from the terminal's working directory.
3. Use an isolated Python environment. Imports across the archive include `numpy`, `opencv-python`, `matplotlib`, `pandas`, `scipy`, `scikit-image`, `pytesseract`, and `Pillow`. Writing Excel files also needs a pandas-compatible Excel writer such as `openpyxl`. The original package versions are not recorded.
4. For OCR-based scripts, install the separate Tesseract application and set `pytesseract.pytesseract.tesseract_cmd` to its actual executable path.
5. Inspect the chosen script's input path, calibration values, and output filename. Work on copies when rerunning analyses, since exports can replace an existing file with the same name.
6. Run the script in an environment that supports interactive plot windows. Some variants request manual scale-bar selection.

Before treating a rerun as a reproduction, confirm the exact script, input images, processing settings, and calibration used for the original result. For example, the 2500× calibration is `10 / 242` micrometers per pixel in `neat_scatterall.py` and `10 / 244` in `scatterall.py` and `final/improving.py`. The historical files have not been changed to resolve that difference.
