import cv2
import numpy as np
import exiftool
import math
import os
import tkinter as tk

from glob import glob
from tqdm import tqdm

from datetime import datetime

from typing import Dict, Union, Tuple

#TODO: open only one file each time

class Photo:
    def __init__(self, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        self.fn = filename
        self.md = None
        self.im = None

    def __lt__(self, other):
        return self.get_date() < other.get_date()

    def save(self, filename: str):
        cv2.imwrite(filename, self.im)
        print(f"\nImage saved: {filename}")

    def show(self, image=None):
        margin = 50
        if image is None:
            image = self.im
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
        if self.im is None:
            self.im = cv2.imread(self.fn)
            if self.im is None:
                raise FileNotFoundError(self.fn)
        return self.im

    def get_metadata(self) -> Dict[str, Union[str, float]]:
        if self.md is None:
            with exiftool.ExifToolHelper() as et:
                # -G -> Group, -n -> numeric values
                self.md = et.get_metadata(self.fn, params=["-G", "-n"])[0]
        return self.md

    def size(self) -> Tuple[int, int]:
        metadata = self.get_metadata()
        w = metadata.get('File:ImageWidth')
        h = metadata.get('File:ImageHeight')
        return w, h

    def get_brand(self) -> str:
        return self.get_metadata()["EXIF:Make"]

    def get_date(self) -> datetime:
        date_str = self.get_metadata()['EXIF:DateTimeOriginal']
        return datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')

    def get_roll_angle(self) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        # Try to get angles. Tags may change depending on brand.
        if 'OLYMPUS' in brand.upper():
            return metadata.get('MakerNotes:RollAngle') or metadata.get('Composite:RollAngle')

    def get_pitch_angle(self) -> float:
        metadata = self.get_metadata()
        brand = self.get_brand()
        # Try to get angles. Tags may change depending on brand.
        if 'OLYMPUS' in brand.upper():
            return metadata.get('MakerNotes:PitchAngle') or metadata.get('Composite:PitchAngle')

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

    def rotate(self, angle_degrees, crop=True):
        """ Rotate and cut to remove black sides """
        if angle_degrees == 0:
            return self.load_image()

        h, w = self.load_image().shape[:2]
        center_x, center_y = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle_degrees, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2) - center_x
        M[1, 2] += (new_h / 2) - center_y

        rotated_image = cv2.warpAffine(self.load_image(), M, (new_w, new_h))

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
        return rotated_image

    def align(self, target_roll: float = 0.0, crop: bool = True) -> np.ndarray:
        """
        Aign image for target Roll angle.
        Args:
            target_roll (float): Target rotation angle (Roll)
        Returns:
            np.ndarray: The resultant image
        """
        #print(f"Processing: {self.fn}")

        current_roll = self.get_roll_angle()
        #print(f"  - Current Roll angle: {current_roll:.2f}°")

        # Calculate needed rotation
        rotation_to_apply = target_roll - current_roll
        #print(f"  - Needed rotation to {target_roll}°: {rotation_to_apply:.2f}°")

        if abs(rotation_to_apply) < 0.1:
            #print("  - Image already aligned. No action is needed.")
            return self.load_image()

        # Rotates and crops image
        self.load_image()
        self.im = self.rotate(rotation_to_apply, crop)
        #print("  - Success: Image rotated and cropped.")

        return self.im


class PhotoList():
    def __init__(self, input_folder, output_folder, target_orientation=None):
        self.input = input_folder
        self.output = output_folder
        self.target = target_orientation
        self.photos = None

    def save(self):
        for f in self.photos:
            output_path = os.path.join(self.output, os.path.basename(f.fn))
            cv2.imwrite(output_path, f.im)

    def timelapse(self, filename):
        h, w, _ = self.photos[0].im.shape
        size = (w, h)
        FPS = 5

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filename, fourcc, FPS, size)

        if not writer.isOpened():
            print("ERROR: Writer could not be started. Check the codec and the write permissions.")
            return

        print(f"Creating '{filename}' com {FPS} FPS...")
        for f in tqdm(self.photos, desc="Processing frames"):
            writer.write(f.im)

        writer.release()
        print("\n-------------------------------------")
        print(f"Vídeo saved as: {os.path.abspath(filename)}")
        print("-------------------------------------")

    def read(self):
        if self.photos is None:
            self.photos = sorted([Photo(f)
                                  for f in glob(os.path.join(self.input, '*.JPG'))
                                  ])

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
        """ Align all photos. """
        if not os.path.exists(self.output):
            os.makedirs(self.output)

        self.read()

        # O alvo para o ângulo de Roll (horizonte)
        TARGET_ROLL_ANGLE = 0.0

        print(f"Iniciando alinhamento 2D para {len(self.photos)} fotos...")

        for f in tqdm(self.photos):
            f.align(TARGET_ROLL_ANGLE, crop=False)

        print("\nAlinhamento concluído!")


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
    pl = PhotoList('originais', 'corrigidas', TARGET_ORIENTATION)
    pl.align()
    # pl.save()
    pl.timelapse('video.mp4')
