import os
import json
import glob
import numpy as np


# optional: progress bars
# try:
#     from tqdm import tqdm
# except ImportError:
#     tqdm = None

EXECUTABLE_FLAG = 0x20000000  # IMAGE_SCN_MEM_EXECUTE



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



def parse_one_sample(s):
    """
    Parse a SINGLE EMBER-style JSON object (Python dict)
    and return (row, label).
    """

    g = s.get("general", {}) or {}
    strs = s.get("strings", {}) or {}
    sec = s.get("section", {}) or {}
    sections = sec.get("sections", []) or []
    head = s.get("header", {}) or {}
    coff = head.get("coff", {}) or {}
    opt = head.get("optional", {}) or {}
    datadir = s.get("datadirectories", []) or []

    # imports / exports
    imports = s.get("imports", {}) or {}
    num_DLL = len(imports)
    tot_imports = sum(len(x) for x in imports)
    import_DLL_ratio = (tot_imports / num_DLL) if num_DLL > 0 else 0.0

    exports = s.get("exports", []) or []
    tot_exports = len(exports)

    if len(datadir) < 3:
        rsrctbl = {}
    else:
        rsrctbl = datadir[2] or {}

    # base feature row
    row = [
        g.get("size", 0),

        num_DLL,
        tot_imports,
        import_DLL_ratio,
        tot_exports,

        len(sections),

        strs.get("numstrings", 0),
        float(strs.get("avlength", 0.0)),
        strs.get("printables", 0),
        strs.get("entropy", 0),

        coff.get("timestamp", 0),
        opt.get("sizeof_headers", 0),
    ]

    # sections
    row += _section_data(sections)

    # byte histogram stats
    row += _byte_hist_stats(s.get("histogram", []))

    # if you want byte-entropy stats, also add:
    # row += _byte_entropy_stats(s.get("byteentropy", []))

    features = np.array(row, dtype=np.float32)

    return features

