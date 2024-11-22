import os
import numpy as np
import random
import pathlib

class Shuffler:
    def __init__(self, array: np.ndarray) -> None:
        self.original = array
        self.shuffled = self.original.copy()
        self.x = self.original.shape[1]
        self.y = self.original.shape[0]
        self._pieces = []

    def _split(self, x: int, y: int, x_list: list, y_list: list) -> None:
        if len(self._pieces) > 0:
            self._pieces.clear()

        for x_n, x_piece in enumerate(x_list):
            for y_n, y_piece in enumerate(y_list):
                self._pieces.append(
                    self.original[y_n * y:y_piece, x_n * x:x_piece]
                )

    def _generate_image(self, cols: int) -> None:
        chunks = [
            np.vstack(chunk) for chunk in zip(*[iter(self._pieces)] * cols)
        ]
        self.shuffled = np.hstack(np.array(chunks, dtype=np.uint8))

    def shuffle(self, matrix: tuple) -> np.ndarray:
        x = int(self.x / matrix[0])
        x_list = list(range(x, self.x + 1, x))

        y = int(self.y / matrix[1])
        y_list = list(range(y, self.y + 1, y))

        self._split(x, y, x_list, y_list)
        random.shuffle(self._pieces)

        self._generate_image(matrix[1])
        return self.shuffled

def shuffle_npy_files(src_folder: str, dst_folder: str, matrix: tuple) -> None:
    total_files = sum([len(files) for r, d, files in os.walk(src_folder) if any(f.endswith('.npy') for f in files)])
    processed_files = 0

    for root, _, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.npy'):
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(
                    dst_folder, 
                    os.path.relpath(root, src_folder), 
                    file
                )
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                
                array = np.load(src_file_path)
                shuffler = Shuffler(array)
                shuffled_array = shuffler.shuffle(matrix)
                np.save(dst_file_path, shuffled_array)

                processed_files += 1
                print(f'Processed {processed_files}/{total_files} files')

src_folder = 'data/top_5_compressed'
dst_folder = 'data_shuffle/top_5_compressed'
matrix = (4, 4)

shuffle_npy_files(src_folder, dst_folder, matrix)
