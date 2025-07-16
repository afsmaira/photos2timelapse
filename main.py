import shutil

import cv2
import numpy as np
import exiftool
import math
import os
import tkinter as tk

from glob import glob
from tqdm import tqdm

from datetime import datetime

from typing import Dict, Union, Tuple, List, Optional, Iterable


def for_gen(l: Iterable, desc: str, verbose: bool) -> Iterable:
    return tqdm(l, desc=desc) \
            if verbose else l

class Photo:
    def __init__(self, filename: str, out_folder: str, in_ram: bool = False, verbose: bool = False):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        self.fn = filename
        self.out = os.path.join(out_folder, os.path.basename(self.fn))
        self.md = None
        self.im = None
        self.in_ram = in_ram
        self.date = None
        self.verbose = verbose

    def __lt__(self, other):
        return self.get_date() < other.get_date()

    def save(self, filename: str):
        if self.in_ram:
            cv2.imwrite(filename, self.im)
        else:
            shutil.copyfile(self.out, filename)
        print(f"\nImage saved: {filename}")

    def show(self, image=None):
        margin = 50
        if image is None:
            image = self.im \
                    if self.in_ram \
                    else cv2.imread(self.fn or self.out)
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        img_h, img_w = image.shape[:2]

        screen_ratio = (screen_width - margin) / (screen_height - margin)
        img_ratio = img_w / img_h

        if img_ratio > screen_ratio:
            new_w = screen_width - margin
            new_h = int(new_w / img_ratio)
        else:
            new_h = screen_height - margin
            new_w = int(new_h * img_ratio)
        resized_img = cv2.resize(image, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)
        cv2.imshow('Processed image', resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_image(self):
        if self.im is not None:
            return self.im
        fn = self.fn or self.out
        self.fn = None
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

    def size(self) -> Tuple[int, int]:
        metadata = self.get_metadata()
        w = int(metadata.get('File:ImageWidth', 0))
        h = int(metadata.get('File:ImageHeight', 0))
        return w, h

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
        w, h = self.size()

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
        if self.verbose:
            print(f"Processing: {self.fn}")

        current_roll = self.get_roll_angle()
        if self.verbose:
            print(f"  - Current Roll angle: {current_roll:.2f}°")

        # Calculate needed rotation
        rotation_to_apply = target_roll - current_roll
        if self.verbose:
            print(f"  - Needed rotation to {target_roll}°: {rotation_to_apply:.2f}°")

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
        f_px = (focal / 36.0) * max(h, w)
        dx = int(f_px * math.tan(d_pitch_rad))
        M_translation = np.float32([[1, 0, dx], [0, 1, 0]])
        translated_image = cv2.warpAffine(image,
                                          M_translation,
                                          dsize=(w, h),
                                          borderMode=cv2.BORDER_CONSTANT,
                                          borderValue=[0,0,0])

        if self.in_ram:
            self.im = translated_image
        else:
            cv2.imwrite(self.out, translated_image)


class PhotoList():
    def __init__(self, input_folder: str, output_folder: str, beg_date: datetime = None, end_date: datetime = None, target_orientation=None, verbose: bool = True):
        self.input = input_folder
        self.output = output_folder
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        self.target = target_orientation
        self.photos: Optional[List[Photo]] = None
        self.verbose: bool = verbose
        if beg_date is None:
            beg_date = datetime.min
        if end_date is None:
            end_date = datetime.max
        self.date_interval: Tuple[datetime, datetime] = beg_date, end_date

    def __iter__(self):
        self.read()
        return iter(self.photos)

    def __len__(self):
        return len(self.photos)

    def max_dimensions(self):
        max_w = max_h = 0
        for photo in for_gen(self.photos, "Finding max dimensions",
                             self.verbose):
            img = photo.load_image()
            h, w = img.shape[:2]
            max_w = max(max_w, w)
            max_h = max(max_h, h)
        return max_w, max_h

    def save(self):
        for f in for_gen(self.photos, 'Saving', self.verbose):
            output_path = os.path.join(self.output, os.path.basename(f.fn))
            f.save(output_path)

    def pad_all(self, color: Tuple[int] = (0, 0, 0)) -> Tuple[int, int]:
        max_w, max_h = self.max_dimensions()
        for photo in for_gen(self.photos, "Applying padding",
                             self.verbose):
            photo.pad(max_w, max_h, color)
        return max_w, max_h

    def timelapse(self, filename):
        size = self.pad_all()
        FPS = 15

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, FPS, size)

        if not writer.isOpened():
            print("ERROR: Writer could not be started. Check the codec and the write permissions.")
            return

        if self.verbose:
            print(f"Creating '{filename}' with {FPS} FPS...")
        for f in for_gen(self.photos, "Making video", self.verbose):
            img = f.load_image()
            writer.write(img)

        writer.release()
        print("\n-------------------------------------")
        print(f"Vídeo saved as: {os.path.abspath(filename)}")
        print("-------------------------------------")

    def read(self):
        if self.photos is None:
            self.photos = []
            for f in for_gen(glob(os.path.join(self.input, '*.JPG')),
                             'Reading files', self.verbose):
                p = Photo(f, self.output)
                if self.date_interval[0] <= p.get_date() <= self.date_interval[1]:
                    self.photos.append(p)
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

    def align(self):
        """ Align all photos """
        self.read()
        target_roll = 0.0

        for f in for_gen(self.photos, 'Aligning images angle',
                         self.verbose):
            f.align(target_roll, crop=False)

        pitches = map(lambda f: f.get_pitch_angle(), self.photos)
        pitches = list(filter(lambda p: p is not None, pitches))
        target_pitch = sum(pitches) / len(pitches)

        # Assuming small angle oscillation
        for f in for_gen(self.photos, 'Aligning images position',
                         self.verbose):
            f.shift(target_pitch)


if __name__ == '__main__':
    # p = Photo('test.jpg')
    # TARGET_ROLL_ANGLE = 0.0
    # p.align(TARGET_ROLL_ANGLE, False)
    # p.save('aligned.jpg')
    # p.show()

    TARGET_ORIENTATION = {
        "roll": 0.0,
        "pitch": None,
        "yaw": None
    }
    pl = PhotoList('originais', 'corrigidas',
                   target_orientation=TARGET_ORIENTATION,
                   end_date=datetime(2025, 4, 16))
    pl.align()
    pl.timelapse('video.mp4')
