"""
Manual annotation tool for calibration points.

The tool uses the 3D template file to guide point-by-point annotation on 2D
images. Each run handles one way and the three corresponding images
(`cam0/cam1/cam2`) in sequence.
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from typing import Dict

print("detect_board.py: starting", flush=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

try:
    from .lib.visualize_coords_from_template import parse_template
except ImportError:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    if _script_dir not in sys.path:
        sys.path.insert(0, _script_dir)
    from lib.visualize_coords_from_template import parse_template


def _log(msg: str) -> None:
    print(msg, flush=True)


_log("detect_board.py: imports OK")


class ManualAnnotator:
    """Interactive manual annotation tool."""

    def __init__(self, image_path, template_path, output_path):
        self.image_path = image_path
        self.template_path = template_path
        self.output_path = output_path

        self.image = plt.imread(image_path)
        sections = parse_template(template_path)

        self.point_list = []
        self.point_labels = []

        if "CORNERS" in sections and sections["CORNERS"][0].size > 0:
            coords, names = sections["CORNERS"]
            for coord, name in zip(coords, names):
                self.point_list.append(coord)
                self.point_labels.append(f"CORNER: {name}")

        for key in sorted(sections.keys()):
            if key.startswith("BEADS_WAY") and sections[key][0].size > 0:
                coords, names = sections[key]
                for coord, name in zip(coords, names):
                    self.point_list.append(coord)
                    self.point_labels.append(f"BEAD: {name}")

        self.total_points = len(self.point_list)
        print(f"\nTotal points to annotate: {self.total_points}")
        print(f"  - Corners: {sum(1 for l in self.point_labels if 'CORNER' in l)}")
        print(f"  - Beads: {sum(1 for l in self.point_labels if 'BEAD' in l)}")

        self.annotated_points = [np.nan] * self.total_points
        self.current_idx = 0
        self.saved = False
        self.zoom_level = 1.0
        self.zoom_center = None
        self.xlim = None
        self.ylim = None

        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.update_display()

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def update_display(self):
        """Refresh the interactive display."""
        self.ax.clear()
        self.ax.imshow(self.image)

        if self.xlim is not None and self.ylim is not None:
            self.ax.set_xlim(self.xlim)
            self.ax.set_ylim(self.ylim)
        else:
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()

        if self.current_idx < self.total_points:
            progress = f"{self.current_idx + 1}/{self.total_points}"
            label = self.point_labels[self.current_idx]
            annotated_count = sum(1 for p in self.annotated_points if not np.isnan(p).any())
            title = (
                f"Progress: {progress} | Annotated: {annotated_count}/{self.total_points} | "
                f"Current: {label}\n"
            )
            title_color = "blue"
        else:
            annotated_count = sum(1 for p in self.annotated_points if not np.isnan(p).any())
            title = f"All Points Done! | Annotated: {annotated_count}/{self.total_points}\n"
            title_color = "green"

        title += (
            "Left click: annotate | Right click: delete previous | "
            "'n': skip | 's': save | 'q': quit | Scroll: zoom | 'r': reset"
        )
        self.ax.set_title(title, fontsize=11, color=title_color)
        self.ax.axis("off")

        for i, (point_2d, label) in enumerate(zip(self.annotated_points, self.point_labels)):
            if not np.isnan(point_2d).any():
                x, y = point_2d
                color, marker, size = ("black", "x", 15) if "CORNER" in label else ("tab:blue", ".", 8)
                self.ax.plot(x, y, marker=marker, color=color, markersize=size, markeredgewidth=2)

                if i == self.current_idx and self.current_idx < self.total_points:
                    circle = Circle((x, y), radius=20, color="yellow", fill=False, linewidth=3)
                    self.ax.add_patch(circle)

        if self.current_idx < self.total_points and not np.isnan(self.annotated_points[self.current_idx]).any():
            x, y = self.annotated_points[self.current_idx]
            circle = Circle((x, y), radius=25, color="cyan", fill=False, linewidth=4)
            self.ax.add_patch(circle)

        self.fig.canvas.draw()

    def on_click(self, event):
        """Handle mouse-click events."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:
            if self.current_idx >= self.total_points:
                return

            x, y = int(event.xdata), int(event.ydata)
            self.annotated_points[self.current_idx] = np.array([x, y])
            print(f"[{self.current_idx + 1}/{self.total_points}] {self.point_labels[self.current_idx]}: ({x}, {y})")

            self.current_idx += 1
            if self.current_idx >= self.total_points:
                print("\nAll points annotated! Auto-saving...")
                self.update_display()
                self.save_points()
                self.saved = True
                print("Auto-save completed! Press 'q' to close.")
            else:
                self.update_display()

        elif event.button == 3:
            if self.current_idx > 0:
                self.current_idx -= 1
                self.annotated_points[self.current_idx] = np.nan
                print(f"Deleted point {self.current_idx + 1}: {self.point_labels[self.current_idx]}")
                self.update_display()

    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == "s":
            self.save_points()
            self.saved = True
            plt.close(self.fig)
        elif event.key == "q":
            print("Quit without saving")
            self.saved = False
            plt.close(self.fig)
        elif event.key == "n":
            if self.current_idx >= self.total_points:
                return
            self.annotated_points[self.current_idx] = np.array([np.nan, np.nan])
            print(f"Skipped point {self.current_idx + 1}: {self.point_labels[self.current_idx]}")
            self.current_idx += 1
            if self.current_idx >= self.total_points:
                print("\nAll points processed! Auto-saving...")
                self.update_display()
                self.save_points()
                self.saved = True
                print("Auto-save completed! Press 'q' to close.")
            else:
                self.update_display()
        elif event.key == "left":
            if self.current_idx > 0:
                self.current_idx -= 1
                self.update_display()
        elif event.key == "right":
            if self.current_idx < self.total_points - 1:
                self.current_idx += 1
                self.update_display()
        elif event.key in {"+", "="}:
            self._zoom_at_center(1.2)
        elif event.key == "-":
            self._zoom_at_center(1.0 / 1.2)
        elif event.key == "r":
            self.xlim = None
            self.ylim = None
            self.zoom_level = 1.0
            self.zoom_center = None
            self.update_display()

    def on_scroll(self, event):
        """Zoom with the mouse wheel."""
        if event.inaxes != self.ax:
            return

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return

        zoom_factor = 1.1 if event.button == "up" else 1.0 / 1.1

        if self.xlim is None or self.ylim is None:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = self.xlim
            ylim = self.ylim

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor

        self.xlim = (xdata - new_x_range / 2, xdata + new_x_range / 2)
        self.ylim = (ydata - new_y_range / 2, ydata + new_y_range / 2)
        self.zoom_center = (xdata, ydata)
        self.zoom_level *= zoom_factor
        self.update_display()

    def _zoom_at_center(self, zoom_factor):
        """Zoom around the current view center."""
        if self.xlim is None or self.ylim is None:
            self.xlim = self.ax.get_xlim()
            self.ylim = self.ax.get_ylim()

        xlim = self.xlim
        ylim = self.ylim

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        new_x_range = x_range / zoom_factor
        new_y_range = y_range / zoom_factor

        self.xlim = (x_center - new_x_range / 2, x_center + new_x_range / 2)
        self.ylim = (y_center - new_y_range / 2, y_center + new_y_range / 2)
        self.zoom_level *= zoom_factor
        self.update_display()

    def save_points(self):
        """Save the annotated coordinates and a preview image."""
        valid_indices = [i for i, p in enumerate(self.annotated_points) if not np.isnan(p).any()]
        skipped = len(self.annotated_points) - len(valid_indices)

        if not valid_indices:
            print("[WARN] No points to save")
            return

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w") as f:
            f.write("# Manual annotation file\n")
            f.write("# Format: index x y X Y Z label\n")
            f.write("# index: point index in template (0-based)\n")
            f.write("# x y: 2D image coordinates (pixels)\n")
            f.write("# X Y Z: 3D world coordinates (mm) from template\n")
            f.write("# label: point label\n")
            f.write("#\n")

            for idx in valid_indices:
                point_2d = self.annotated_points[idx]
                point_3d = self.point_list[idx]
                label = self.point_labels[idx]
                x, y = point_2d
                X, Y, Z = point_3d
                f.write(f"{idx} {int(x)} {int(y)} {X:.6f} {Y:.6f} {Z:.6f} {label}\n")

        print(f"\nSaved {len(valid_indices)} points to: {self.output_path}")
        if skipped > 0:
            print(f"Skipped {skipped} point(s)")

        output_dir = os.path.dirname(self.output_path)
        marking_dir = os.path.join(output_dir, "manual_marking")
        os.makedirs(marking_dir, exist_ok=True)

        image_basename = os.path.basename(self.image_path)
        name, _ext = os.path.splitext(image_basename)
        output_image_path = os.path.join(marking_dir, f"{name}_manual_marking.png")

        self.fig.savefig(output_image_path, dpi=200, bbox_inches="tight", format="png")
        print(f"Saved annotated image to: {output_image_path}")


def _find_template_file(calibration_folder: str, way_name: str) -> str:
    """Find the template file for the specified way."""
    calibration_raw_dir = os.path.dirname(os.path.normpath(calibration_folder))

    possible_paths = [
        os.path.join(calibration_raw_dir, f"cali_coords_template_{way_name}.txt"),
        os.path.join(calibration_raw_dir, f"cali_coords_template_{way_name.upper()}.txt"),
        os.path.join(calibration_folder, f"cali_coords_template_{way_name}.txt"),
        os.path.join(calibration_folder, f"Calibration_Coords_Template_{way_name.upper()}.txt"),
    ]

    for template_path in possible_paths:
        if os.path.exists(template_path):
            return template_path

    for search_dir in [calibration_raw_dir, calibration_folder]:
        templates = glob.glob(os.path.join(search_dir, f"*coords*template*{way_name}*.txt"))
        if templates:
            return templates[0]
        templates = glob.glob(os.path.join(search_dir, f"*{way_name}*template*.txt"))
        if templates:
            return templates[0]

    raise FileNotFoundError(
        f"Template file not found for {way_name}. "
        f"Searched in {calibration_raw_dir} and {calibration_folder}. "
        f"Please make sure cali_coords_template_{way_name}.txt exists."
    )


def _organize_images_by_way(calibration_folder: str) -> Dict[str, Dict[str, str]]:
    """Group images by way and camera."""
    image_exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
    all_images = []
    for ext in image_exts:
        all_images.extend(glob.glob(os.path.join(calibration_folder, ext)))

    organized: Dict[str, Dict[str, str]] = {}
    folder_basename = os.path.basename(calibration_folder).lower()

    for img_path in all_images:
        basename = os.path.basename(img_path).lower()
        way_match = re.search(r"way(\d+)", basename) or re.search(r"way(\d+)", folder_basename)
        if not way_match:
            continue
        way_name = f"way{way_match.group(1)}"

        cam_name = None
        if "cam0" in basename or "top" in basename:
            cam_name = "cam0"
        elif "cam1" in basename or "side1" in basename:
            cam_name = "cam1"
        elif "cam2" in basename or "side2" in basename:
            cam_name = "cam2"

        if cam_name is None:
            continue

        organized.setdefault(way_name, {})
        if cam_name not in organized[way_name]:
            organized[way_name][cam_name] = img_path

    return organized


def batch_annotate(calibration_folder: str, output_root: str | None = None):
    """
    Batch annotation entry point: process one way and its three images
    (`cam0/cam1/cam2`) in sequence.
    """
    calibration_folder = os.path.normpath(calibration_folder)

    folder_name = os.path.basename(calibration_folder)
    date_str = datetime.now().strftime("%Y%m%d")

    if "cali" in folder_name.lower():
        session_name = f"cali_{date_str}"
    else:
        session_name = f"{folder_name}_{date_str}"

    print("Scanning images...")
    organized = _organize_images_by_way(calibration_folder)

    if not organized:
        raise ValueError(f"No images found in {calibration_folder} (way1/way2/cam0/cam1/cam2)")

    print(f"\nFound {len(organized)} way group(s):")
    for way_name, cams in organized.items():
        print(f"  {way_name}: {len(cams)} camera(s) ({', '.join(cams.keys())})")
    print(f"Session name: {session_name}")

    if output_root is not None:
        results_root = os.path.normpath(output_root)
    else:
        calibration_folder_abs = os.path.abspath(calibration_folder)
        if "calibration_raw" in calibration_folder_abs:
            parts = calibration_folder_abs.split(os.sep)
            calibration_raw_idx = None
            for i, part in enumerate(parts):
                if part == "calibration_raw":
                    calibration_raw_idx = i
                    break

            if calibration_raw_idx is not None:
                parent_dir = os.sep.join(parts[:calibration_raw_idx])
                results_root = os.path.join(parent_dir, "calibration_results", session_name)
            else:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                results_root = os.path.join(project_root, "data", "calibration_results", session_name)
        else:
            parent_dir = os.path.dirname(calibration_folder_abs)
            if os.path.basename(parent_dir) == "calibration_raw":
                grandparent_dir = os.path.dirname(parent_dir)
                results_root = os.path.join(grandparent_dir, "calibration_results", session_name)
            else:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                results_root = os.path.join(project_root, "data", "calibration_results", session_name)

    os.makedirs(results_root, exist_ok=True)
    print(f"Results will be saved to: {results_root}")

    way_names = sorted(organized.keys(), key=lambda x: int(re.search(r"way(\d+)", x).group(1)))
    way_name = way_names[0]
    cams = organized[way_name]
    print(f"\n{'=' * 70}")
    print(f"Processing {way_name.upper()}")
    print(f"{'=' * 70}")

    template_path = _find_template_file(calibration_folder, way_name)
    print(f"Using template: {os.path.basename(template_path)}")

    cam_order = ["cam0", "cam1", "cam2"]
    for cam_name in cam_order:
        if cam_name not in cams:
            print(f"\nSkipping {cam_name} (image not found)")
            continue

        image_path = cams[cam_name]
        print(f"\nAnnotating {cam_name}: {os.path.basename(image_path)}")

        output_path = os.path.join(results_root, f"cali_{way_name}_{cam_name}_image_coords.txt")
        print(f"Output: {output_path}")

        annotator = ManualAnnotator(image_path, template_path, output_path)
        plt.show()

        if annotator.saved:
            print(f"[OK] {cam_name} annotation saved")
        else:
            print(f"[SKIP] {cam_name} annotation cancelled")
            print("Stopping batch processing...")
            return

    print(f"\n{'=' * 70}")
    print("Batch annotation completed!")
    print(f"Results saved to: {results_root}")
    print(f"{'=' * 70}")


def main():
    _log("detect_board.py: main() started")
    parser = argparse.ArgumentParser(
        description="Manually annotate calibration points for one way across cam0/cam1/cam2"
    )
    parser.add_argument("--folder", type=str, required=True,
                        help="Calibration image folder path (for example cali_xxx_way1_pic)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional coordinate-output folder. If omitted it is inferred automatically.",
    )

    args = parser.parse_args()
    try:
        batch_annotate(args.folder, output_root=args.output)
    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
