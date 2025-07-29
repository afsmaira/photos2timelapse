import numpy as np
import exiftool
import math
import os
import tkinter as tk
import shutil
import argparse
import logging

import cv2

from moviepy import VideoFileClip

from glob import glob
from tqdm import tqdm

from datetime import datetime

from typing import Dict, Union, Tuple, List, Optional, Iterable

from concurrent.futures import ProcessPoolExecutor, as_completed

Number = Union[int, float]
JSON = Dict[str, float | int | str | bool | None | dict | list]

ERROR = 0
INFO = 1
DEBUG = 2

logger = logging.getLogger('TimelapseApp')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log.txt', mode='w', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)


def screen_dimensions() -> Tuple[int, int]:
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height


def for_gen(l: Iterable, desc: str, total: int = None, verbose: int = INFO) -> Iterable:
    return tqdm(l, desc=desc, total=total) \
        if verbose == INFO else l


class Photo:
    def __init__(self, filename: str, out_folder: str, in_ram: bool = False, verbose: int = INFO):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        self.fn = filename
        self.out = os.path.join(out_folder, os.path.basename(self.fn))
        self.md = None
        self.im = None
        self.in_ram = in_ram
        self.date = None
        self.get_date()
        self.verbose = verbose

    def __lt__(self, other):
        return self.get_date() < other.get_date()

    def filename(self):
        return self.fn or self.out

    def basename(self):
        return os.path.basename(self.filename())

    def save(self, filename: str):
        if self.in_ram:
            cv2.imwrite(filename, self.im)
        else:
            shutil.copyfile(self.out, filename)
        if self.verbose == DEBUG:
            print(f"\nImage saved: {filename}")

    @staticmethod
    def fit_screen(image):
        margin = 50
        screen_width, screen_height = screen_dimensions()
        screen_ratio = (screen_width - margin) / (screen_height - margin)

        img_h, img_w = image.shape[:2]
        img_ratio = img_w / img_h

        if img_ratio > screen_ratio:
            new_w = screen_width - margin
            new_h = int(new_w / img_ratio)
        else:
            new_h = screen_height - margin
            new_w = int(new_h * img_ratio)
        return cv2.resize(image, (new_w, new_h),
                          interpolation=cv2.INTER_AREA)

    def show(self, image=None):
        if image is None:
            image = self.im \
                    if self.in_ram \
                    else cv2.imread(self.fn or self.out)
        cv2.imshow('Processed image', self.fit_screen(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def orig2out(self, erase_fn: bool = False) -> str:
        """ Copy original file to output folder """
        if self.fn is not None and self.out is not None:
            shutil.copyfile(self.fn, self.out)
            if erase_fn:
                self.fn = None
        return self.filename()

    def load_image(self):
        if self.im is not None:
            return self.im
        fn = self.orig2out(True)
        image = cv2.imread(fn)
        if image is None:
            raise FileNotFoundError(fn)
        if self.in_ram:
            self.im = image.copy()
        return image

    def get_metadata(self) -> Dict[str, Union[str, float]]:
        if self.md is None:
            with exiftool.ExifToolHelper() as et:
                # -G -> Group, -n -> numeric values
                self.md = et.get_metadata(self.fn, params=["-G", "-n"])[0]
        return self.md

    def shape(self, metadata: bool = True) -> Tuple[int, int]:
        if metadata:
            self.get_metadata()
            w = int(self.md.get('File:ImageWidth', 0))
            h = int(self.md.get('File:ImageHeight', 0))
            return h, w
        return tuple(self.load_image().shape[:2])

    @staticmethod
    def image_shape(photo):
        return photo.shape(False)

    def get_brand(self) -> str:
        return self.get_metadata()["EXIF:Make"]

    def get_date(self) -> datetime:
        if self.date is None:
            date_str = self.get_metadata()['EXIF:DateTimeOriginal']
            self.date = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
        return self.date

    def get_roll_angle(self) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        # Try to get angles. Tags may change depending on brand.
        if 'OLYMPUS' in brand.upper():
            return metadata.get('MakerNotes:RollAngle') or metadata.get('Composite:RollAngle') or 0

    def get_pitch_angle(self) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        # Try to get angles. Tags may change depending on brand.
        if 'OLYMPUS' in brand.upper():
            return metadata.get('MakerNotes:PitchAngle') or metadata.get('Composite:PitchAngle') or 0

    def get_yam_angle(self) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        # Try to get angles. Tags may change depending on brand.
        if 'OLYMPUS' in brand.upper():
            return metadata.get('EXIF:GPSImgDirection')

    def get_focal_length(self) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        # Try to get angles. Tags may change depending on brand.
        if 'OLYMPUS' in brand.upper():
            return metadata.get('EXIF:FocalLengthIn35mmFormat') or metadata.get('Composite:FocalLength35efl')

    def get_orientation(self) -> Dict[str, float]:
        """ Orientation metadata """
        roll = self.get_roll_angle()
        pitch = self.get_pitch_angle()
        yaw = self.get_yam_angle()
        focal = self.get_focal_length()
        h, w = self.shape()

        if any(v is None for v in [roll, pitch, yaw, focal]):
            raise ValueError(f"Values not found for {self.fn}")

        return {
            "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
            "focal_length": float(focal),
            "width": w, "height": h
        }

    def pad(self, target_w: int, target_h: int, color: Tuple[int] = (0, 0, 0)):
        img = self.load_image()
        h, w = img.shape[:2]

        dw = target_w - w
        dh = target_h - h
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        padded_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        if self.in_ram:
            self.im = padded_image
        else:
            cv2.imwrite(self.out, padded_image)

    def rotate(self, angle_degrees, crop=True):
        """ Rotate """
        image = self.load_image()

        h, w = image.shape[:2]
        center_x, center_y = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center_x
        M[1, 2] += (new_h / 2) - center_y

        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

        if crop:
            # The greatest rectangle inside the rotated image
            orig_rect = np.array([
                [0, 0], [w, 0], [w, h], [0, h]
            ], dtype=np.float32)

            rotated_rect_corners = cv2.transform(np.array([orig_rect]), M)[0]

            x_coords = [p[0] for p in rotated_rect_corners]
            y_coords = [p[1] for p in rotated_rect_corners]
            angle_rad = math.radians(angle_degrees)
            sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)

            if w * cos_a > h * sin_a:
                x = (w * cos_a - h * sin_a) / 2
                y = (w * sin_a + h * cos_a) / 2
                crop_w = int(w - 2 * x)
                crop_h = int(h - 2 * x * (sin_a/cos_a))
            else:
                x = (h * cos_a - w * sin_a) / 2
                y = (h * sin_a + w * cos_a) / 2
                crop_w = int(w-2*x*(cos_a/sin_a))
                crop_h = int(h - 2*x)

            # Rotated image center
            center_crop_x, center_crop_y = new_w / 2, new_h / 2

            # Cropped area
            x1 = int(center_crop_x - crop_w / 2)
            x2 = int(center_crop_x + crop_w / 2)
            y1 = int(center_crop_y - crop_h / 2)
            y2 = int(center_crop_y + crop_h / 2)

            rotated_image = rotated_image[y1:y2, x1:x2]
        if self.in_ram:
            self.im = rotated_image
        else:
            cv2.imwrite(self.out, rotated_image)

    def align(self, target_roll: float = 0.0, crop: bool = True):
        """
        Aign image for target Roll angle.
        Args:
            target_roll (float): Target rotation angle (Roll)
        Returns:
            np.ndarray: The resultant image
        """
        current_roll = self.get_roll_angle()

        # Calculate needed rotation
        rotation_to_apply = target_roll - current_roll

        if abs(rotation_to_apply) < 0.1:
            return

        self.rotate(rotation_to_apply, crop)

    def shift(self, target_pitch: float):
        image = self.load_image()
        h, w = image.shape[:2]

        orientation = self.get_orientation()
        current_pitch = orientation['pitch']
        focal = orientation['focal_length']

        if abs(current_pitch - target_pitch) < 0.1:
            return

        d_pitch_deg = target_pitch - current_pitch
        d_pitch_rad = math.radians(d_pitch_deg)
        f_p = focal / 36.0
        f_px = f_p * max(h, w)
        f_py = f_p * min(h, w)
        dx = int(f_px * math.tan(d_pitch_rad)) * abs(math.sin(self.get_roll_angle()))
        dy = int(f_py * math.tan(d_pitch_rad)) * abs(math.cos(self.get_roll_angle()))
        M_translation = np.float32([[1, 0, dx], [0, 1, dy]])
        translated_image = cv2.warpAffine(image,
                                          M_translation,
                                          dsize=(w, h),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=[0, 0, 0])

        if self.in_ram:
            self.im = translated_image
        else:
            cv2.imwrite(self.out, translated_image)


class PhotoList:
    def __init__(self, input_folder: str, output_folder: str, video_fn: str = 'video.mp4', fps: int = 15,
                 beg_date: datetime = None, end_date: datetime = None, target_orientation=None,
                 verbose_level: int = INFO):
        self.input = input_folder
        self.output = output_folder
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.target = target_orientation
        self.photos: Optional[List[Photo]] = None
        self.verbose: int = verbose_level
        if beg_date is None:
            beg_date = datetime.min
        if end_date is None:
            end_date = datetime.max
        self.date_interval: Tuple[datetime, datetime] = beg_date, end_date
        self.fn = video_fn
        self.fps = fps

    def __iter__(self):
        self.read()
        return iter(self.photos)

    def __len__(self):
        return len(self.photos)

    def to_out(self):
        self.read()
        for f in self.photos:
            f.orig2out(True)

    def max_dimensions(self):
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(photo.shape, False)
                       for photo in self.photos]

            shapes = [future.result()
                      for future in for_gen(as_completed(futures),
                                            total=len(futures),
                                            desc='Processing files',
                                            verbose=self.verbose)]
        max_w = max_h = 0
        for h, w in shapes:
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        return max_w, max_h

    def save(self):
        for f in for_gen(self.photos, 'Saving', self.verbose):
            output_path = os.path.join(self.output, os.path.basename(f.fn))
            f.save(output_path)

    def pad_all(self, color: Tuple[int] = (0, 0, 0)) -> Tuple[int, int]:
        max_w, max_h = self.max_dimensions()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(photo.pad, max_w, max_h, color)
                       for photo in self.photos]
            for _ in for_gen(futures, "Applying padding",
                             self.verbose): pass

        return max_w, max_h

    def timelapse(self, overwrite: bool = False):
        filename = self.fn
        h0, w0 = self.photos[0].shape(False)

        if os.path.exists(filename):
            if overwrite:
                os.remove(filename)
            else:
                raise FileExistsError(f'File {filename} already exists!')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, self.fps, (w0, h0))

        if not writer.isOpened():
            print("ERROR: Writer could not be started. Check the codec and the write permissions.")
            return

        if self.verbose == DEBUG:
            print(f"Creating '{filename}' with {self.fps} FPS...")
        for f in for_gen(self.photos, "Making video", self.verbose):
            img = f.load_image()
            writer.write(img)

        writer.release()
        print("\n-------------------------------------")
        print(f"Vídeo saved as: {os.path.abspath(filename)}")
        print("-------------------------------------")

    def to_whatsapp(self, suffix: str = "_wapp"):
        base, ext = os.path.splitext(self.fn)
        output_video_path = f"{base}{suffix}{ext}"

        if self.verbose == DEBUG:
            print(f"\nIniciando conversão para WhatsApp (via MoviePy): {output_video_path}")

        try:
            clip = VideoFileClip(self.fn)

            target_width = 720
            if clip.w > target_width:
                clip = clip.resized(width=target_width)

            clip.write_videofile(
                output_video_path,
                fps=clip.fps,
                codec="libx264",
                preset="medium",
                ffmpeg_params=["-crf", "28", "-b:a", "128k"],
                audio_codec="aac"
            )

            clip.close()

        except Exception as e:
            print(f"ERROR during MoviePy convertion: {e}")
            print("Be sure that FFmpeg is installed and in system PATH, and that the input video is not corrupted.")

    def read(self):
        if self.photos is None:
            files_list = glob(os.path.join(self.input, '*.JPG'))
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = [executor.submit(Photo, path, self.output, False, self.verbose)
                           for path in files_list]

                self.photos = [future.result()
                               for future in for_gen(as_completed(futures),
                                                     total=len(files_list),
                                                     desc='Processing files',
                                                     verbose=self.verbose)
                               if self.date_interval[0] <= future.result().get_date() <= self.date_interval[1]]
            self.photos.sort()

    def get_camera_matrix(self, focal_length, width, height):
        """ Camera Matrix """
        f_px = (focal_length / 36) * width
        return np.array([
            [f_px, 0, width / 2],
            [0, f_px, height / 2],
            [0, 0, 1]
        ], dtype=np.float32)

    def rotation_matrix(self, roll, pitch, yaw):
        """Cria uma matriz de rotação 3D a partir dos ângulos de Euler."""
        r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _align_photo_worker(photo_obj: Photo, target_roll: float, crop: bool):
        photo_obj.align(target_roll, crop)
        return photo_obj.out

    def align(self):
        """ Align all photos """
        self.to_out()
        target_roll = 0.0

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            n = len(self.photos)
            for _ in for_gen(executor.map(
                    PhotoList._align_photo_worker, self.photos,
                    [target_roll] * n,
                    [False] * n
            ), total=n, desc='Aligning images angle',
                    verbose=self.verbose):
                pass

        self.pad_all()
        self.stabilize()

    def stabilize(self, ref_idx: int = 0):
        self.read()

        ref_photo = self.photos[ref_idx]
        ref_img = ref_photo.load_image()
        display_img = Photo.fit_screen(ref_img)
        img_h, img_w = ref_img.shape[:2]
        display_h, display_w = display_img.shape[:2]

        r = cv2.selectROI("Select the Region of Interest (ROI) for stabilizing", display_img, fromCenter=False, showCrosshair=False)
        roi_x, roi_y, roi_w, roi_h = map(int, r)

        scale_x = img_w / display_w
        scale_y = img_h / display_h

        roi_x = int(roi_x * scale_x)
        roi_y = int(roi_y * scale_y)
        roi_w = int(roi_w * scale_x)
        roi_h = int(roi_h * scale_y)

        if roi_w == 0 or roi_h == 0:
            raise Exception('Invalid ROI')

        # cv2.goodFeaturesToTrack: detects Shi-Tomasi corners
        # maxCorners: maximum number of corners
        # qualityLevel: minimum corner quality
        # minDistance: minimum distance between corners
        feature_params = dict(maxCorners=100, qualityLevel=0.3,
                              minDistance=7, blockSize=7)

        mask = np.zeros_like(ref_img[:, :, 0])
        mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255

        prev_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)

        if p0 is None or len(p0) == 0:
            raise Exception("No point detected in ROI. Try to select area with more details.")

        lk_params = dict(winSize=(15,15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        transforms_x = []
        transforms_y = []

        print("Tracking shift correction...")
        dx = dy = 0
        current_pos_x = current_pos_y = 0

        for i, current_photo in enumerate(for_gen(self.photos, "Tracking features",
                                                  self.verbose)):
            current_img = current_photo.load_image()
            current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)

            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 5:
            try:
                p.set_description(f'Tracking ({current_photo.filename()} {len(good_new)}/{MIN_GOOD_POINTS})')
            except:
                pass

                #print(f"Warn: Low tracked points in frame {i}. Recalibrating...")
                p0 = cv2.goodFeaturesToTrack(current_gray, mask=None, **feature_params)
                if p0 is None or len(p0) == 0:
                    print(f"Error: Low tracked points again in frame {i}.")
                    dx, dy = 0, 0
                else:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    if len(good_new) < 5:
                        dx, dy = 0, 0
                    else:
                        dx = np.mean(good_new[:,0] - good_old[:,0])
                        dy = np.mean(good_new[:,1] - good_old[:,1])
            else:
                dx = np.mean(good_new[:,0] - good_old[:,0])
                dy = np.mean(good_new[:,1] - good_old[:,1])

            current_pos_x += dx
            current_pos_y += dy

            M = np.float32([[1, 0, -current_pos_x], [0, 1, -current_pos_y]])

            h, w = current_img.shape[:2]
            processed_img = cv2.warpAffine(current_img, M, (w, h),
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=[0, 0, 0])

            if current_photo.in_ram:
                current_photo.im = processed_img
            else:
                cv2.imwrite(current_photo.out, processed_img)

            prev_gray = current_gray.copy()
            p0 = good_new.reshape(-1,1,2)

        if self.verbose == DEBUG:
            print("\nTracking stabilization finished.")

    def angle_distribution(self, angle_type: str, show: bool = True):
        self.read()
        angles = []
        if angle_type == 'roll':
            getter_method = Photo.get_roll_angle
            angle_label = "Roll angle (degrees)"
        elif angle_type == 'pitch':
            getter_method = Photo.get_pitch_angle
            angle_label = "Pitch angle (degrees)"
        elif angle_type == 'yaw':
            getter_method = Photo.get_yam_angle
            angle_label = "Yaw angle (degrees)"
        else:
            return

        for photo in for_gen(self.photos, f"Getting {angle_type} angles", total=len(self.photos), verbose=self.verbose):
            angle = getter_method(photo)
            if angle is not None:
                try:
                    angles.append(angle * 180 / np.pi)
                except (ValueError, TypeError):
                    pass

        if show:
            plt.figure(figsize=(10, 6))
            min_angle, max_angle = min(angles), max(angles)
            num_bins = max(10, int((max_angle - min_angle) / 1.0))
            plt.hist(angles, bins=num_bins, edgecolor='black', alpha=0.7)
            plt.xlim(min_angle - 5, max_angle + 5)
            plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='0 Graus')
            plt.legend()

            plt.title(f'{angle_label.capitalize()} distribution')
            plt.xlabel(angle_label)
            plt.ylabel('Photos number')
            plt.grid(axis='y', alpha=0.75)
            plt.tight_layout()
            plt.show()

        return angles


def process_input():
    parser = argparse.ArgumentParser(
        description="Makes a timelapse video from a sequence of photos."
    )

    parser.set_defaults(verbose_level=1)

    parser.add_argument(
        '-q', '--quiet',
        action='store_const',
        const=0,
        dest='verbose_level',
        help='Verbosity level ERROR (0), shows only errors'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_const',
        const=2,
        dest='verbose_level',
        help='Verbosity level DEBUG (2), shows every steps'
    )

    parser.add_argument(
        '-i', '--input',
        dest='input',
        type=str,
        help='Input files folder (.JPG)',
        default='input'
    )

    parser.add_argument(
        '-f', '--fixed',
        dest='fixed',
        type=str,
        help='Folder for fixed image files. Default: fixed',
        default='fixed'
    )

    parser.add_argument(
        '-o', '--output',
        dest='output',
        type=str,
        help='Output file',
        default='video.mp4'
    )

    parser.add_argument(
        '--fps',
        dest='fps',
        type=int,
        default=15,
        help='Output file FPS. Default: 15'
    )

    parser.add_argument(
        '--start-date',
        dest='start_date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime.min,
        help='Filter for beginning date (YYYY-MM-DD). Default: 0001-01-01'
    )

    parser.add_argument(
        '--end-date',
        dest='end_date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        default=datetime.max,
        help='Filter for end date (YYYY-MM-DD). Default: 9999-12-31'
    )

    parser.add_argument(
        '--roll',
        dest='roll',
        type=float,
        default=0.0,
        help='Target roll angle in degrees. Default: 0.0'
    )

    args = parser.parse_args()

    TARGET_ORIENTATION = {
        "roll": args.roll,
        "pitch": None,
        "yaw": None
    }
    pl = PhotoList(input_folder=args.input,
                   output_folder=args.fixed,
                   target_orientation=TARGET_ORIENTATION,
                   beg_date=args.start_date,
                   end_date=args.end_date,
                   video_fn=args.output,
                   fps=args.fps,
                   verbose_level=args.verbose_level)
    pl.align()
    pl.timelapse(overwrite=True)
    pl.to_whatsapp()
