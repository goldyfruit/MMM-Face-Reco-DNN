import os
from typing import Iterator

import cv2
import numpy as np


class Image:
    """Image transforms and dataset-file enumeration shared by the tools."""

    image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    @staticmethod
    def resize(
        image: np.ndarray,
        width: int | None = None,
        height: int | None = None,
        inter: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """Resize while preserving aspect ratio, preferring width when supplied."""
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        for value in (width, height):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError("Resize dimensions must be positive integers")
        if h <= 0 or w <= 0:
            raise ValueError("Cannot resize an empty image")

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (max(1, int(w * r)), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, max(1, int(h * r)))

        if dim == (w, h):
            return image

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized

    @staticmethod
    def adjust_brightness_contrast(
        image: np.ndarray, brightness: float = 0, contrast: float = 0
    ) -> np.ndarray:
        """Apply the existing linear adjustment, avoiding a neutral full-frame copy."""
        brightness = float(brightness)
        contrast = float(contrast)
        if not np.isfinite(brightness) or not np.isfinite(contrast):
            raise ValueError("Brightness and contrast must be finite")
        if brightness == 0 and contrast == 0:
            return image
        beta = 0
        # See the OpenCV docs for more info on the `beta` parameter to addWeighted
        # https://docs.opencv.org/3.4.2/d2/de8/group__core__array.html#gafafb2513349db3bcff51f54ee5592a19
        return cv2.addWeighted(
            image,
            1 + float(contrast) / 100.0,
            image,
            beta,
            float(brightness),
        )

    @staticmethod
    def list_images(basePath: str, contains: str | None = None) -> Iterator[str]:
        """Yield recognized image files, optionally filtering their names."""
        # return the set of files that are valid
        return Image.list_files(
            basePath, validExts=Image.image_types, contains=contains
        )

    @staticmethod
    def list_files(
        basePath: str,
        validExts: tuple[str, ...] | None = None,
        contains: str | None = None,
    ) -> Iterator[str]:
        """Yield files beneath a directory with optional extension/name filters."""
        # loop over the directory structure
        for rootDir, dirNames, filenames in os.walk(basePath):
            # loop over the filenames in the current directory
            for filename in filenames:
                # if the contains string is not none and the filename does not contain
                # the supplied string, then ignore the file
                if contains is not None and filename.find(contains) == -1:
                    continue

                # determine the file extension of the current file
                ext = filename[filename.rfind(".") :].lower()

                # check to see if the file is an image and should be processed
                if validExts is None or ext.endswith(validExts):
                    # construct the path to the image and yield it
                    imagePath = os.path.join(rootDir, filename)
                    yield imagePath
