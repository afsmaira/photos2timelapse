import numpy as np
import exiftool
import math
import os
import tkinter as tk
import shutil
import argparse
import logging

import matplotlib.pyplot as plt

# Remove QFactoryLoader verbose
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.*=false;qt.core.*=false;*.debug=false;*.info=false'

# Remove QElfParser verbose
os.environ['LD_DEBUG'] = ''
os.environ['QT_DEBUG_PLUGINS'] = '0'
os.environ['G_MESSAGES_DEBUG'] = ''

import cv2

from moviepy import VideoFileClip

from glob import glob
from tqdm import tqdm

from datetime import datetime

from typing import Dict, Union, Tuple, List, Optional, Iterable, Set

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
    if total is None:
        try:
            total = len(l)
        except Exception as err:
            logger.error(f'for_gen() - {err}')
    return tqdm(l, desc=desc, total=total) \
        if verbose == INFO else l


def match_template(curr: np.array, template: np.array, method: str | int, verbose: int = INFO) -> Tuple[bool, int, int, float, np.array]:
    if method == 'div':
        # extremelly slow. TODO: parallelize
        template_original_float = template.astype(np.float32)
        template_original_float += 1e-6

        h_template, w_template = template.shape[:2]
        h_current, w_current = curr.shape[:2]

        min_ = (float('inf'), 0, 0)

        if verbose == DEBUG:
            print("Starting Template Matching (ratio and variance)...")

        result = None
        for y in range(h_current - h_template + 1):
            for x in range(w_current - w_template + 1):
                patch = curr[y:y + h_template, x:x + w_template].copy()
                patch_float = patch.astype(np.float32)
                result = patch_float / template_original_float
                std_dev = np.std(result)
                min_ = min(min_, (std_dev, x, y))
        std_dev, x, y = min_
        sim = 1.0 - std_dev / 20
        ok = result is not None
    else:
        result = cv2.matchTemplate(curr, template, method)
        ok = result is not None
        if not ok:
            return False, -1, -1, -1, None
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            x, y = min_loc
        else:
            x, y = max_loc

        sim = max_val if method not in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED] else (1.0 - min_val)
    return ok, x, y, sim, result


def debug_stabilize(curr_photo, template: np.array, result: np.array,
                    x: int, y: int, sim: float):
    titles = ["1. Original Template",
              "2. Current image with Match region",
              "3. Similarity map"]

    curr = curr_photo.load_image()
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    display_template = Photo.fit_screen(template, 0.25)
    cv2.imshow(titles[0], display_template)
    cv2.moveWindow(titles[0], 0, 0)

    display_current_img_debug = Photo.fit_screen(curr.copy(), 0.5)
    debug_img_h, debug_img_w = display_current_img_debug.shape[:2]
    img_h, img_w = curr.shape[:2]

    scale_factor_x_debug = debug_img_w / img_w
    scale_factor_y_debug = debug_img_h / img_h

    scaled_top_left = (int(x * scale_factor_x_debug),
                       int(y * scale_factor_y_debug))

    scaled_bottom_right = (int(scaled_top_left[0] + template_gray.shape[1] * scale_factor_x_debug),
                           int(scaled_top_left[1] + template_gray.shape[0] * scale_factor_y_debug))

    cv2.rectangle(display_current_img_debug, scaled_top_left, scaled_bottom_right, (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    bg_color = (0, 0, 0)
    font_color = (255, 255, 255)
    line_type = cv2.LINE_AA

    text_pos_filename = (10, 50)
    text_pos_similarity = (10, 100)

    text_filename = f"File: {curr_photo.basename()}"
    text_similarity = f"Similarity: {sim}"

    cv2.putText(display_current_img_debug, text_filename, text_pos_filename, font, font_scale, bg_color, thickness + 2, line_type)
    cv2.putText(display_current_img_debug, text_filename, text_pos_filename, font, font_scale, font_color, thickness, line_type)

    cv2.putText(display_current_img_debug, text_similarity, text_pos_similarity, font, font_scale, bg_color, thickness + 2, line_type)
    cv2.putText(display_current_img_debug, text_similarity, text_pos_similarity, font, font_scale, font_color, thickness, line_type)

    cv2.imshow(titles[1], display_current_img_debug)
    cv2.moveWindow(titles[1], display_template.shape[1] + 10, 0)

    vis_result = cv2.normalize(result, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    vis_result_display = Photo.fit_screen(vis_result, 0.25)

    cv2.imshow(titles[2], vis_result_display)
    cv2.moveWindow(titles[2], display_template.shape[1] + display_current_img_debug.shape[1] + 20, 0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    def fit_screen(image, width: float = 1):
        margin = 50
        screen_width, screen_height = screen_dimensions()
        screen_ratio = (int(width * screen_width) - margin) / (screen_height - margin)

        img_h, img_w = image.shape[:2]
        img_ratio = img_w / img_h

        if img_ratio > screen_ratio:
            new_w = int(width * screen_width) - margin
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

    def get_angle(self, angle_type: str) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        angle = None
        degrees = True  # Change for brands it is given in radians in next lines
        if 'OLYMPUS' in brand.upper():
            if angle_type == 'roll':
                angle = metadata.get('MakerNotes:RollAngle') or metadata.get('Composite:RollAngle') or 0
            elif angle_type == 'pitch':
                angle = metadata.get('MakerNotes:PitchAngle') or metadata.get('Composite:PitchAngle') or 0
            elif angle_type == 'yaw':
                angle = metadata.get('EXIF:GPSImgDirection')
        if isinstance(angle, str):
            angle = float(angle)
        if degrees and isinstance(angle, (int, float)):
            angle *= np.pi / 180
        return angle

    def get_roll_angle(self) -> float:
        return self.get_angle('roll')

    def get_pitch_angle(self) -> float:
        return self.get_angle('pitch')

    def get_yam_angle(self) -> float:
        return self.get_angle('yaw')

    def get_focal_length(self, f35: bool = False) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        f = None
        if f35:
            if 'OLYMPUS' in brand.upper():
                f = metadata.get('EXIF:FocalLengthIn35mmFormat') or metadata.get('Composite:FocalLength35efl')
        else:
            f = metadata.get('EXIF:FocalLength')
            if f is None:
                f = self.get_focal_length(f35=True) / metadata.get('Composite:ScaleFactor35efl')
                if f is None:
                    raise ValueError("Focal length real não encontrada nos metadados.")
        if isinstance(f, str):
            f = float(f)
        return f

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
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "focal_length": focal,
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

    def cam_matrix(self):
        md = self.get_metadata()
        fl_mm = self.get_focal_length()
        sensor_d = md.get('MakerNotes:FocalPlaneDiagonal')
        model = md.get('EXIF:Model')
        if model == 'TG-6':
            sensor_w, sensor_h = 6.17, 4.55
            if abs(sensor_d - np.sqrt(sensor_w ** 2 + sensor_h ** 2)) > 0.2:
                raise ValueError('Sensor dimensions not found')
        else:
            raise Exception('Unknown camera model')
        width_px, height_px = md.get('File:ImageWidth'), md.get('File:ImageHeight')
        cx, cy = width_px // 2, height_px // 2
        fx = fl_mm * (width_px / sensor_w)
        fy = fl_mm * (height_px / sensor_h)
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def rot_matrix(angle_type: str, angle: float) -> np.ndarray:
        if angle_type == 'pitch':
            return np.array([[1, 0, 0],
                             [0, np.cos(angle), -np.sin(angle)],
                             [0, np.sin(angle), np.cos(angle)]])
        if angle_type == 'yaw':
            return np.array([[np.cos(angle), 0, np.sin(angle)],
                             [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        if angle_type == 'roll':
            return np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0],
                             [0, 0, 1]])

    def homo_matrix(self, angle_type: str, angle: float) -> np.ndarray:
        K = self.cam_matrix()
        R = self.rot_matrix(angle_type, angle)
        return K @ R @ np.linalg.inv(K)

    def crop_borders(self):
        image = self.load_image()
        h, w = image.shape[:2]

        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        non_black_coords = cv2.findNonZero(gray_image)

        if non_black_coords is not None:
            x_coords = non_black_coords[:, 0, 0]
            y_coords = non_black_coords[:, 0, 1]

            min_x = np.min(x_coords)
            max_x = np.max(x_coords)
            min_y = np.min(y_coords)
            max_y = np.max(y_coords)

            cropped_image = image[min_y:max_y+1, min_x:max_x+1]
        else:
            cropped_image = np.zeros_like(image)

        if self.in_ram:
            self.im = cropped_image
        else:
            cv2.imwrite(self.filename(), cropped_image)

    def fix_angle(self, angle_type: str, target: float, eps: float = 0.1):
        curr = self.get_angle(angle_type)
        rotation_to_apply = target - curr

        if abs(rotation_to_apply) < eps:
            return

        image = self.load_image()
        h, w = image.shape[:2]
        if angle_type == 'roll':
            self.rotate(rotation_to_apply)
        else:
            new_image = cv2.warpPerspective(image,
                                            self.homo_matrix(angle_type, rotation_to_apply),
                                            (3 * w, h),
                                            borderMode=cv2.BORDER_CONSTANT)

            if self.in_ram:
                self.im = new_image
            else:
                cv2.imwrite(self.filename(), new_image)
        self.crop_borders()

    def fix_pitch(self, target_pitch: float):
        self.fix_angle('pitch', target_pitch)

    def fix_yaw(self, target_yaw: float):
        self.fix_angle('yaw', target_yaw)

    def fix_roll(self, target_roll: float = 0.0):
        self.fix_angle('roll', target_roll)

    def rotate(self, angle: float):
        angle_degrees = angle * 180 / np.pi
        image = self.load_image()

        h, w = image.shape[:2]
        center_x, center_y = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((center_x, center_y),
                                    angle_degrees, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center_x
        M[1, 2] += (new_h / 2) - center_y

        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))

        if self.in_ram:
            self.im = rotated_image
        else:
            cv2.imwrite(self.filename(), rotated_image)

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
    def __init__(self, input_folder: str, output_folder: str, target_orientation: Dict[str, Number] = None,
                 video_fn: str = 'video.mp4', fps: int = 15, outliers: bool = False,
                 beg_date: datetime = None, end_date: datetime = None, excluded: List[str] = None,
                 tracking_params: Dict[str, Number] = None, labels: str = '', verbose_level: int = INFO,
                 stabilize_method: str = 'corners', stabilize_debug: bool = False,
                 similarity_limit: float = 0.0):
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

    def __getitem__(self, item):
        return self.photos[item]

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
                                            desc='Measuring images',
                                            verbose=self.verbose)]
        max_w = max_h = 0
        for h, w in shapes:
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        return max_w, max_h

    def save(self):
        for f in for_gen(self.photos, 'Saving', verbose=self.verbose):
            output_path = os.path.join(self.output, os.path.basename(f.fn))
            f.save(output_path)

    def pad_all(self, color: Tuple[int] = (0, 0, 0)) -> Tuple[int, int]:
        max_w, max_h = self.max_dimensions()
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(photo.pad, max_w, max_h, color)
                       for photo in self.photos]
            for _ in for_gen(futures, "Applying padding",
                             verbose=self.verbose): pass

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
        for f in for_gen(self.photos, "Making video", verbose=self.verbose):
            img = f.load_image()

            if self.labels == 'datetime':
                text = f.get_date().strftime("%Y-%m-%d %H:%M:%S")
            elif self.labels == 'date':
                text = f.get_date().strftime("%Y-%m-%d")
            elif self.labels == 'time':
                text = f.get_date().strftime("%H:%M:%S")
            elif self.labels == 'file':
                text = f.basename()
            else:
                text = None

            if text is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                bg_color = (0, 0, 0)
                font_color = (255, 255, 255)
                position = (10, img.shape[:2][0] - 30)
                line_type = cv2.LINE_AA

                bg_thickness = 100
                bg_color = (0, 0, 0)

                cv2.putText(img, text, (position[0] - bg_thickness // 2, position[1] + bg_thickness // 2), font,
                            font_scale, bg_color, thickness + 2, line_type)
                cv2.putText(img, text, position, font, font_scale, font_color, thickness, line_type)

                cv2.putText(img, text, (position[0] - bg_thickness // 2, position[1] + bg_thickness // 2), font,
                            font_scale, bg_color, thickness + 2, line_type)
                cv2.putText(img, text, position, font, font_scale, font_color, thickness, line_type)

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
                                                     desc='Reading files',
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

    def remove_outliers(self, angle_type: str = None, apply: bool = True):
        if angle_type is None:
            to_rm: Set[Photo] = set()
            for angle_type in ['yaw', 'pitch', 'roll']:
                to_rm.union(self.remove_outliers(angle_type, False))
            for f in to_rm:
                self.photos.remove(f)
        if self.outliers > 0:
            target_type = self.target.get(angle_type, 'median')
            n = len(self.photos)
            angles = [(f.get_angle(angle_type), f) for f in self.photos]
            if target_type == 'avg':
                target = sum(a for a, _ in angles) / n
            elif target_type == 'median':
                target = list(sorted(angles))[(n - 1) // 2][0]
            else:
                target = float(target_type)
            angles = sorted([(abs(a - target), a, f)
                             for a, f in angles])
            if apply:
                for _, _, f in angles[int(n*(1-self.outliers)):]:
                    self.photos.remove(f)
            return set(f for _, _, f in angles[int(n*(1-self.outliers)):])

    def fix_angle(self, angle_type: str):
        target_type = self.target.get(angle_type, 'median')
        n = len(self.photos)
        angles = [f.get_angle(angle_type) for f in self.photos]
        if target_type == 'avg':
            target = sum(angles) / n
        elif target_type == 'median':
            target = list(sorted(angles))[(n - 1) // 2]
        else:
            target = float(target_type)
        n = len(self.photos)

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for _ in for_gen(executor.map(
                    PhotoList._align_photo_worker, self.photos,
                    [angle_type] * n,
                    [target] * n
            ), total=n, desc=f'Aligning {angle_type} angle',
                    verbose=self.verbose):
                pass

        self.pad_all()

    def fix_roll(self):
        self.fix_angle('roll')

    def fix_pitch(self):
        self.fix_angle('pitch')

    def fix_yaw(self):
        self.fix_angle('yaw')

    def align(self):
        self.to_out()
        self.remove_outliers()
        self.fix_pitch()
        self.fix_roll()
        self.stabilize()

    def stabilize(self, ref_idx: int = 0):
        self.read()

        ref_photo = self.photos[ref_idx]
        ref_img = ref_photo.load_image()
        display_img = Photo.fit_screen(ref_img)
        img_h, img_w = ref_img.shape[:2]
        img_r = img_w / img_h
        display_h, display_w = display_img.shape[:2]
        display_r = display_w / display_h

        r = cv2.selectROI("Select the Region of Interest (ROI) for stabilizing",
                          display_img, fromCenter=False, showCrosshair=False)
        cv2.destroyAllWindows()
        roi_x, roi_y, roi_w, roi_h = map(int, r)

        scale_x = img_w / display_w
        scale_y = img_h / display_h

        roi_x = int(roi_x * scale_x)
        roi_y = int(roi_y * scale_y)
        roi_w = int(roi_w * scale_x)
        roi_h = int(roi_h * scale_y)

        if roi_w == 0 or roi_h == 0:
            raise Exception('Invalid ROI')

        if self.stabilize_method == 'corners':
            default_params = dict(maxCorners=100, qualityLevel=0.9,
                                  minDistance=7, blockSize=7)
            feature_params = dict() if self.track_params is None else self.track_params
            for k, v in default_params.items():
                if k not in feature_params:
                    feature_params[k] = default_params[k]

            curr_roi = ref_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
            mask = np.zeros_like(ref_img[:, :, 0])
            mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = 255

            prev_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **feature_params)

            if p0 is None or len(p0) == 0:
                raise Exception("No point detected in ROI. Try to select area with more details.")

            MIN_GOOD_POINTS_FRAC = 0.95
            MIN_GOOD_POINTS = MIN_GOOD_POINTS_FRAC * len(p0)

            lk_params = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            if self.verbose == DEBUG:
                print("Tracking shift correction...")

            transforms_x = [0]
            transforms_y = [0]
            M = np.float32([[1, 0, 0], [0, 1, 0]])

            for i, current_photo in enumerate(p := for_gen(self.photos, "Tracking features",
                                                           verbose=self.verbose)):
                current_img = current_photo.load_image()
                if current_img is None:
                    if self.verbose in [ERROR, DEBUG]:
                        print(f"ERROR: Image {current_photo.filename()} could not be loaded. Skipping frame.")
                    continue

                current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

                dx, dy = 0, 0
                good_new_count_display = 0

                if i > 0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)

                    good_new = np.array([]) if p1 is None else p1[st == 1]
                    good_new_count_display = len(good_new)

                    if p1 is None or len(good_new) < MIN_GOOD_POINTS:
                        if self.verbose == DEBUG:
                            print(f"Warn: Low tracked points in frame {i}. Recalibrating...")
                        dx, dy = 0, 0
                        # dx, dy = -transforms_x[-1], -transforms_y[-1]
                    else:
                        good_old = p0[st == 1]

                        dx = np.mean(good_new[:, 0] - good_old[:, 0])
                        dy = np.mean(good_new[:, 1] - good_old[:, 1])

                    transforms_x.append(dx)
                    transforms_y.append(dy)

                    try:
                        p.set_description(
                            f'Tracking ({current_photo.basename()})')
                    except:
                        pass

                    logger.info(f'{current_photo.basename()} {good_new_count_display} {MIN_GOOD_POINTS}')

                    total_dx = np.sum(transforms_x)
                    total_dy = np.sum(transforms_y)
                    M = np.float32([[1, 0, -total_dx], [0, 1, -total_dy]])

                h_curr, w_curr = current_img.shape[:2]
                processed_img = cv2.warpAffine(current_img, M, (w_curr, h_curr),
                                               borderMode=cv2.BORDER_CONSTANT,
                                               borderValue=[0, 0, 0])

                if current_photo.in_ram:
                    current_photo.im = processed_img
                else:
                    cv2.imwrite(current_photo.out, processed_img)

                if i > 0 and p1 is not None and len(p1[st == 1]) >= MIN_GOOD_POINTS:
                    prev_gray = current_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)

        elif self.stabilize_method == 'template_match':
            template_match_method = cv2.TM_CCOEFF_NORMED
            template_original = ref_img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
            template_orig_x = roi_x
            template_orig_y = roi_y

            if self.verbose == DEBUG:
                print("Template Matching shift correction...")

            to_rm = []

            # TODO: parallize
            for i, current_photo in enumerate(pbar := for_gen(self.photos, "Matching templates",
                                                              verbose=self.verbose)):
                current_img = current_photo.load_image()
                if current_img is None:
                    if self.verbose in [ERROR, DEBUG]:
                        print(f"ERROR: Image {current_photo.basename()} could not be loaded. Skipping frame.")
                    continue

                current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
                template_gray = cv2.cvtColor(template_original, cv2.COLOR_BGR2GRAY)

                if template_gray.shape[0] > current_gray.shape[0] or template_gray.shape[1] > current_gray.shape[1]:
                    processed_img = current_img.copy()
                else:
                    ok, x, y, sim, result = match_template(current_gray, template_gray, template_match_method)

                    M = np.float32([[1, 0, template_orig_x-x],
                                    [0, 1, template_orig_y-y]])

                    h_curr, w_curr = current_img.shape[:2]
                    processed_img = cv2.warpAffine(current_img, M, (w_curr, h_curr),
                                                   borderMode=cv2.BORDER_CONSTANT,
                                                   borderValue=[0, 0, 0])

                    if result is not None and self.stabilize_debug:
                        debug_stabilize(current_photo, template_original, result,
                                        x, y, sim)

                    if sim < self.similarity_limit:
                        to_rm.append(self.photos[i])

                try:
                    pbar.set_description(f'Matching ({current_photo.basename()} sim:{sim:.2f})')
                except Exception as e:
                    pass

                if current_photo.in_ram:
                    current_photo.im = processed_img
                else:
                    cv2.imwrite(current_photo.out, processed_img)

            for photo in to_rm:
                self.photos.remove(photo)

        if self.verbose == DEBUG:
            print("\nTracking stabilization finished.")

    def angle_distribution(self, angle_type: str, show: bool = True):
        self.read()
        self.remove_outliers(angle_type)
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
            plt.axvline(0, color='red', linestyle='dashed', linewidth=1.5, label='0º')
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
        type=str,
        default='median',
        help='Target roll angle in degrees. Default: "median"'
    )

    parser.add_argument(
        '--yaw',
        dest='yaw',
        type=str,
        default='median',
        help='Target yaw angle in degrees. Default: "median"'
    )

    parser.add_argument(
        '--pitch',
        dest='pitch',
        type=str,
        default='median',
        help='Target pitch angle in degrees. Default: "median"'
    )

    parser.add_argument(
        '-l', '--labels',
        dest='labels',
        type=str,
        default='',
        help='Text to be shown in video. If "date", show images dates. If "time", show images times. If "datetime", show images dates and times. If "file", show images filenames. Default: ""'
    )

    parser.add_argument(
        '-x', '--exclude',
        action='append',
        dest='excluded',
        help='File to be excluded from video. This can be used multiple times.'
    )

    parser.add_argument(
        '--corners-level',
        type=float,
        default=0.3,
        dest='corners_level',
        help='Tracking corners quality level. Default: 0.3'
    )

    parser.add_argument(
        '--outliers',
        type=float,
        dest='outliers',
        default=0.0,
        help='Fraction of angles to be considered outliers to be removed. Default: 0.0'
    )

    parser.add_argument(
        '--stabilize',
        type=str,
        dest='stabilize_method',
        default='template_match',
        help='Method used to stabilize. Default: "template_match"'
    )

    parser.add_argument(
        '--sim-limit',
        type=float,
        dest='sim_limit',
        default=0.0,
        help='Similarity limit for stabilize process. Photos with similarity below this are removed. Default: 0.0'
    )

    TARGET_ORIENTATION = dict(roll=args.roll,
                              pitch=args.pitch,
                              yaw=args.yaw)
    tracking_params = dict(qualityLevel=args.corners_level)
    pl = PhotoList(input_folder=args.input,
                   output_folder=args.fixed,
                   target_orientation=TARGET_ORIENTATION,
                   beg_date=args.start_date,
                   end_date=args.end_date,
                   excluded=args.excluded,
                   video_fn=args.output,
                   fps=args.fps,
                   labels=args.labels,
                   verbose_level=args.verbose_level,
                   tracking_params=tracking_params,
                   outliers=args.outliers,
                   stabilize_method=args.stabilize_method,
                   similarity_limit=args.sim_limit)
    pl.align()
    pl.timelapse(overwrite=True)
    pl.to_whatsapp()
