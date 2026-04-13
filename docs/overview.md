# Overview

This repository is structured as a pipeline with four major stages:

1. Acquisition (multi-camera synchronized recording)
2. Calibration (intrinsics/extrinsics; refraction-aware if underwater)
3. Tracking (SLEAP inference + any identity handling)
4. Behavioral analysis (events, ethograms, summary metrics)

See also:
- [data_layout.md](data_layout.md)
- [script_workflow.md](script_workflow.md)
- [calibration.md](calibration.md)
- [sleap.md](sleap.md)
- [analysis.md](analysis.md)
