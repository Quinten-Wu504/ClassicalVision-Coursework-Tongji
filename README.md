# Classical Computer Vision

Computer Vision coursework at Tongji University.

- **Image pyramids:** Gaussian and Laplacian pyramids, reconstruction, and PSNR/SSIM evaluation on Kodak images.
- **Edge detection:** Canny, Sobel, Prewitt and Laplacian operators, evaluated on BSDS500 with OIS/ODS precision, recall and F1.
- **Segmentation:** watershed and region growing combined with edge detectors, evaluated using Dice, IoU and pixel accuracy. Includes erosion, dilation, opening and closing implementations.

## Run

```bash
pip install -r requirements.txt
python pyrMain.py
python edgeMain.py
python segMain.py
```

Run from the repository root on Windows; the pyramid and edge scripts contain Windows-style paths. The datasets are included at these paths:

| Task | Input paths |
| --- | --- |
| Pyramids | `Dataset/KodakLosslessTrue/` |
| Edges | `Dataset/BSDS500/data/images/test/` and `Dataset/BSDS500/data/groundTruth/test/` (`.mat` annotations) |
| Segmentation | `Dataset/Cats/images/` (`.jpg`/`.jpeg`) and `Dataset/Cats/groundTruth/` (matching basenames with `.png` masks) |

Scripts save images and CSV metrics in each module's `outputs/` directory. Datasets and generated results are included. Dependencies are unpinned; full experiments have not been re-run for this archive.
