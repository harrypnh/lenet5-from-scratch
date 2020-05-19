from __future__ import (division, absolute_import, print_function, unicode_literals)
import gzip
import sys, os, tempfile, logging

if sys.version_info >= (3, ):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse

def download(url, destination = None):
    url_open = urllib2.urlopen(url)
    scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
    filename = os.path.basename(path)
    if not filename:
        filename = "downloaded.file"
    if destination:
        filename = os.path.join(destination, filename)
    with open(filename, "wb") as file:
        meta = url_open.info()
        meta_function = meta.getheaders if hasattr(meta, "getheaders") else meta.get_all
        meta_length = meta_function("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))
        file_size_download = 0
        block_sz = 8192
        while True:
            buffer = url_open.read(block_sz)
            if not buffer:
                break
            file_size_download += len(buffer)
            file.write(buffer)
            status = "{0:16}".format(file_size_download)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_download * 100 / file_size)
            status += chr(13)
            print(status, end = "")
        print()
    return filename

def extract_gz(filename):
    with gzip.open(filename, "rb") as in_file:
        with open(filename[0: -3], "wb") as out_file:
            for line in in_file:
                out_file.write(line)

train_images_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
test_images_url = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
test_labels_url = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
if not os.path.isdir("MNIST"):
    os.mkdir("MNIST")
os.chdir("./MNIST")
print("Downloading MNIST dataset...")
train_images_file = download(train_images_url)
train_labels_file = download(train_labels_url)
test_images_file = download(test_images_url)
test_labels_file = download(test_labels_url)
print("\nExtracting...")
extract_gz("./" + train_images_file)
extract_gz("./" + train_labels_file)
extract_gz("./" + test_images_file)
extract_gz("./" + test_labels_file)
print("\nDeleting .gz files...")
os.remove(train_images_file)
os.remove(train_labels_file)
os.remove(test_images_file)
os.remove(test_labels_file)
print("\nMNIST dataset is stored in the 'MNIST' folder")
