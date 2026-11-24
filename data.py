"""Official RGB data in CIFAR file order. Test labels are opened only by evaluation."""
from pathlib import Path
import hashlib
import json
import pickle
import tarfile
import urllib.request

import numpy as np

ROOT = Path("data")
ARCHIVE = ROOT / "cifar-10-python.tar.gz"
URL = "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
MD5 = "c58f30108f718f92721af3b95e74349a"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def archive_batches(names):
    if file_hash(ARCHIVE, "md5") != MD5:
        raise ValueError("Official CIFAR archive checksum mismatch")
    images, labels = [], []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for name in names:
            item = archive.extractfile("cifar-10-batches-py/" + name)
            batch = pickle.load(item, encoding="bytes")  # only after official MD5
            images.append(batch[b"data"].reshape(-1, 3, 32, 32))
            labels.extend(batch[b"labels"])
    return np.concatenate(images), np.asarray(labels, dtype=np.int64)


def stratify(y):
    rng = np.random.default_rng(42)
    parts = [rng.permutation(np.flatnonzero(y == c)) for c in range(10)]
    return (np.sort(np.concatenate([v[500:] for v in parts])),
            np.sort(np.concatenate([v[:500] for v in parts])))


def train_stats(x, ids):
    sums, squares = np.zeros(x.shape[1]), np.zeros(x.shape[1])
    for batch in np.array_split(ids, max(1, len(ids) // 200)):
        a = x[batch].astype(np.float64) / 255
        sums += a.sum(axis=(0, 2, 3))
        squares += np.square(a).sum(axis=(0, 2, 3))
    count = len(ids) * x.shape[2] * x.shape[3]
    mean = sums / count
    return mean.tolist(), np.sqrt(squares / count - mean ** 2).tolist()


def prepare():
    ROOT.mkdir(exist_ok=True)
    if not ARCHIVE.exists():
        tmp = ARCHIVE.with_suffix(".download")
        urllib.request.urlretrieve(URL, tmp)
        if file_hash(tmp, "md5") != MD5:
            raise ValueError("Incomplete download")
        tmp.replace(ARCHIVE)
    x, y = archive_batches([f"data_batch_{i}" for i in range(1, 6)])
    train, val = stratify(y)
    assert x.shape == (50000, 3, 32, 32) and x.dtype == np.uint8
    assert np.array_equal(np.sort(np.r_[train, val]), np.arange(50000))
    assert np.all(np.bincount(y[train]) == 4500) and np.all(np.bincount(y[val]) == 500)
    mean, std = train_stats(x, train)
    for name, array in dict(images=x, labels=y, train=train, val=val).items():
        np.save(ROOT / (name + ".npy"), array)
    info = dict(source=URL, archive_md5=MD5, archive_sha256=file_hash(ARCHIVE),
                split_seed=42, train_sha256=digest(train), val_sha256=digest(val),
                mean=mean, std=std, train_count=len(train), val_count=len(val),
                ids="zero-based positions in official split, in original batch order")
    json_write("results/split.json", info)
    print(json.dumps(info))


if __name__ == "__main__":
    prepare()
