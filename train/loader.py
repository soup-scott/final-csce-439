import os
import json
import glob
import pandas as pd
import numpy as np


# optional: progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

EXECUTABLE_FLAG = 0x20000000  # IMAGE_SCN_MEM_EXECUTE


# def _coarse_histogram(hist, group_size=8, expected_len=256):
#     # hist is length-256 list; return length-(256/group_size) agg
#     if len(hist) != expected_len:
#         raise ValueError(f"histogram length {len(hist)} != {expected_len}")
#     bins = []
#     for i in range(0, expected_len, group_size):
#         bins.append(sum(hist[i:i+group_size]))
#     return bins



def _section_data(sections):
    """
    Returns:
      [text_entropy, rdata_entropy, data_entropy, reloc_entropy, rsrc_entropy, code_entropy,
       text_size,    rdata_size,    data_size,    reloc_size,    rsrc_size,    code_size]
    Missing sections => 0.0
    """
    namedict = {"text": 0, "rdata": 1, "data": 2, "reloc": 3, "rsrc": 4, "code": 5}
    data = [0.0] * (2 * len(namedict))

    if not sections:
        return data

    for s in sections:
        name = (s.get("name") or "").lower().strip(".")
        if name in namedict:
            idx = namedict[name]
            data[idx] = float(s.get("entropy", 0.0))
            data[len(namedict) + idx] = float(s.get("size", 0.0))

    return data

def _byte_hist_stats(hist):
    """
    Compute statistical features from the 256-bin byte histogram.
    Returns:
      [mean, std, min_count, max_count, kurtosis,
       skew, range_count, zero_count, energy, entropy, gini]
    """
    if not hist:
        return [0.0]*11

    if len(hist) != 256:
        hist = (hist + [0]*256)[:256]

    h = np.asarray(hist, dtype=float)
    total = h.sum()
    min_count = float(h.min())
    max_count = float(h.max())
    zero_count = int((h == 0).sum())
    range_count = max_count - min_count

    if total <= 0:
        return [0.0]*11

    p = h / total
    values = np.arange(256, dtype=float)

    # mean, variance, std
    mean = float((values * p).sum())
    var = float(((values - mean)**2 * p).sum())
    std = float(np.sqrt(var)) if var > 0 else 0.0

    # skewness
    if var > 0:
        skew = float(((values - mean)**3 * p).sum() / (var**1.5))
    else:
        skew = 0.0

    # kurtosis
    if var > 0:
        m4 = float(((values - mean)**4 * p).sum())
        kurt = m4 / (var**2) - 3.0   # excess kurtosis
    else:
        kurt = 0.0

    # energy (L2 norm squared of the histogram pmf)
    energy = float((p*p).sum())

    # Shannon entropy
    entropy = float(-(p[p>0] * np.log2(p[p>0])).sum())

    # Gini impurity (1 - sum(p_i^2))
    gini = 1.0 - float((p*p).sum())

    return [
        mean, std, min_count, max_count, kurt,
        skew, range_count, zero_count, energy, entropy, gini
    ]



def _byte_entropy_stats(be, high_thr=7.0, low_thr=2.0):
    """
    Compute compact stats from the 256-bin local byte entropy histogram.

    Returns:
      [
        mean, std, max,
        high_count, low_count,
        variance_ratio (std/mean),
        zero_count,
      ]
    """
    if not be or len(be) != 256:
        return [0.0]*7

    arr = np.asarray(be, dtype=float)

    mean = float(arr.mean())
    std  = float(arr.std())
    maxv = float(arr.max())

    high_count = int((arr > high_thr).sum())
    low_count  = int((arr < low_thr).sum())
    zero_count = int((arr == 0).sum())

    if mean > 0:
        variance_ratio = float(std / mean)
    else:
        variance_ratio = 0.0

    return [
        mean, std, maxv,
        high_count, low_count,
        variance_ratio, zero_count
    ]




def _parse_jsonl_file(path, X_rows, y, max_samples=-1, show_progress=False):
    """
    Append rows/labels from a single .jsonl file into X_rows / y.
    Respects global max_samples (across all files).
    """
    with open(path, "r", encoding="utf-8") as f:
        # optional per-file progress bar
        iterable = f
        if show_progress and tqdm is not None:
            iterable = tqdm(f, desc=os.path.basename(path), leave=False)

        for line in iterable:
            if 0 <= max_samples == len(y):
                break

            s = json.loads(line)
            if s.get("label", -1) == -1:
                continue

            g = s.get("general", {}) or {}
            strs = s.get("strings", {}) or {}
            sec = s.get("section", {}) or {}
            sections = sec.get("sections", []) or []
            head = s.get("header", {}) or {}
            coff = head.get("coff", {}) or {}
            opt = head.get("optional", {}) or {}
            datadir = s.get("datadirectories", []) or []


            imports = s.get("imports", {}) or {}
            num_DLL = len(imports)
            tot_imports = 0
            for imp in imports:
                tot_imports += len(imp)

            if(num_DLL > 0):
                import_DLL_ratio = tot_imports / num_DLL
            else:
                import_DLL_ratio = 0

            exports = s.get("exports", []) or []
            tot_exports = len(exports)


            if len(datadir) < 3:
                rsrctbl = {}
            else:
                rsrctbl = datadir[2] or {}

            row = [
                # general
                g.get("size", 0),
                g.get("entropy", 0.0),

                # imp/exp
                num_DLL,
                tot_imports,
                import_DLL_ratio,
                tot_exports,

                # sections
                len(sections),
                sec.get("overlay", {}).get("size", 0),
                sec.get("overlay", {}).get("entropy", 0),

                # strings
                strs.get("numstrings", 0),
                float(strs.get("avlength", 0.0)),
                strs.get("printables", 0),
                strs.get("entropy", 0),

                # headers
                coff.get("timestamp", 0),
                opt.get("address_of_entrypoint", 0),
                opt.get("sizeof_headers", 0),


                # tables
                len(datadir),
                len(s.get("pefilewarnings", []))
            ]

            row = row + _section_data(sections)
            
            row += _byte_hist_stats(s.get("histogram", []))

            # row += _byte_entropy_stats(s.get("byteentropy", []))

            X_rows.append(row)
            y.append(int(s["label"]))


def load_ember_jsonl(path, max_samples=-1, recursive=False, show_progress=True):
    """
    Load EMBER-style JSONL features from:
      - a single .jsonl file, OR
      - a directory of .jsonl files.

    Args:
        path: str
            Path to a .jsonl file or to a directory containing .jsonl files.
        max_samples: int
            Maximum total samples to load across all files.
            -1 means no limit.
        recursive: bool
            If True and path is a directory, walk subdirectories too.
        show_progress: bool
            If True, print progress bars (requires tqdm).

    Returns:
        X: pandas.DataFrame
        y: list[int]
    """
    X_rows, y = [], []

    if os.path.isdir(path):
        if recursive:
            files = []
            for root, _, fnames in os.walk(path):
                for fn in fnames:
                    if fn.endswith(".jsonl"):
                        files.append(os.path.join(root, fn))
        else:
            files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    elif os.path.isfile(path):
        files = [path]
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")

    # outer progress bar over files
    file_iter = files
    if show_progress and tqdm is not None:
        file_iter = tqdm(files, desc="JSONL files")

    for fp in file_iter:
        if 0 <= max_samples == len(y):
            break
        _parse_jsonl_file(fp, X_rows, y, max_samples=max_samples, show_progress=show_progress)

    cols = [
        "size", "gen_entropy", "num_DLL", "imports", "import_DLL_ratio", "exports", "num_sections", "overlay_size", "overlay_entropy",
        "num_strings", "avlength", "printables", "string_entropy", "timestamp",
        "address_of_entrypoint", "sizeof_headers", "num_datadirectories", "num_warnings",
        "text_entropy", "rdata_entropy", "data_entropy",
        "reloc_entropy", "rsrc_entropy", "code_entropy",
        "text_size", "rdata_size", "data_size",
        "reloc_size", "rsrc_size", "code_size",
    ]
    cols += [
        "byte_mean", "byte_std", "byte_min", "byte_max", "byte_kurt",
        "byte_skew", "byte_range", "byte_zero_count",
        "byte_energy", "byte_entropy", "byte_gini"
    ]
    # cols += [
    #     "byteent_mean",
    #     "byteent_std",
    #     "byteent_max",
    #     "byteent_high_count",
    #     "byteent_low_count",
    #     "byteent_variance_ratio",
    #     "byteent_zero_count",
    # ]


    # cols += [f"byte{i}" for i in range(32)]
    # cols += [f"byte{i}_entropy" for i in range(32)]

    X = pd.DataFrame(X_rows, columns=cols)
    return X, y
