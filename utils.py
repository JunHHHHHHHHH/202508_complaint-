# utils.py
import hashlib
import os

def file_md5(path, chunk_size=8192):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
