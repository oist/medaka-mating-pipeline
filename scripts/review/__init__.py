"""Human identity / head-tail review: proxy videos, review data, ledger, GUI.

Package layout on disk (one folder per experiment under the review root,
default ``<data_root>/review_packages``)::

    <exp>/proxy/<view>.play.mp4      every ``step``-th source frame (~30 fps)
    <exp>/proxy/<view>.full.mp4      every source frame (frame stepping)
    <exp>/proxy/proxy_manifest.json  written last; its presence means "done"
    <exp>/<round_tag>/review_data.npz, review_manifest.json, machine_risk.csv
    <exp>/<round_tag>/human_events.csv, coverage.csv, session_log.csv

The GUI and the ledger only need numpy / pandas / opencv / PyQt5, so a laptop
can review without the GPU / SLEAP / calibration stack.
"""
