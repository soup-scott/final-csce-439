#!/usr/bin/env python3
"""
Evaluate a malware detector against a corpus of sample files.

Supports two detector modes:
  1) CLI: run an executable per-file (e.g., `./detector sample.bin` -> prints 0/1)
  2) HTTP: POST file bytes to a URL that returns 0/1 in the body

Outputs a CSV with per-file results and prints summary metrics.
"""

from __future__ import annotations
import argparse
import csv
import os
import re
import sys
import time
import json
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------
# Utilities
# ----------------------------

def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def infer_label(path: Path, mal_pattern: str, good_pattern: str) -> Optional[int]:
    """
    Infer label from ancestor directory names. Returns 1 for malware, 0 for goodware, or None if unknown.
    Matching is case-insensitive substring search on directory names.
    """
    mal_re = re.compile(mal_pattern, re.IGNORECASE) if mal_pattern else None
    good_re = re.compile(good_pattern, re.IGNORECASE) if good_pattern else None
    for ancestor in [path.parent] + list(path.parents):
        name = ancestor.name
        if mal_re and mal_re.search(name):
            return 1
        if good_re and good_re.search(name):
            return 0
    return None

def parse_label_from_csv(map_csv: Optional[Path]) -> Dict[str, int]:
    if not map_csv:
        return {}
    mapping: Dict[str, int] = {}
    with map_csv.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        # Expect columns: path,label
        for row in reader:
            p = row.get('path') or row.get('file') or row.get('filepath')
            lbl = row.get('label') or row.get('y') or row.get('ground_truth')
            if p is None or lbl is None:
                continue
            try:
                mapping[os.path.normpath(p)] = int(lbl)
            except ValueError:
                pass
    return mapping


def normalize_prediction(text: str) -> Optional[int]:
    """
    Normalize model output -> {0,1}. Accepts text that may contain whitespace/newlines,
    JSON like {"result": 0}, booleans, or a probability.
    Returns None if cannot parse.
    """
    s = (text or "").strip()
    if s == "": 
        return None

    # Exact strings first
    if s == "0": return 0
    if s == "1": return 1

    low = s.lower()
    if low in {"benign","good","goodware","false"}: return 0
    if low in {"malicious","malware","bad","true"}: return 1

    # Try JSON
    try:
        import json as _json
        obj = _json.loads(s)
        # If dict, look for common keys
        if isinstance(obj, dict):
            for k in ["result","prediction","pred","label","y","is_malware","malicious","score","prob","p"]:
                if k in obj:
                    val = obj[k]
                    # booleans
                    if isinstance(val, bool):
                        return 1 if val else 0
                    # numeric
                    try:
                        fv = float(val)
                        # Integers 0/1 or threshold at 0.5
                        if fv in (0.0,1.0):
                            return int(fv)
                        return 1 if fv >= 0.5 else 0
                    except Exception:
                        pass
                    # string like "0"/"1"/"benign"/"malicious"
                    if isinstance(val, str):
                        return normalize_prediction(val)
            # Fallback: any first int 0/1 value in values
            for v in obj.values():
                if v in (0,1):
                    return int(v)
        # If list/tuple, try the first scalar
        if isinstance(obj, (list,tuple)) and obj:
            return normalize_prediction(str(obj[0]))
    except Exception:
        pass

    # Try to grab the first standalone 0 or 1 in the string
    m = re.search(r"(?:^|)([01])(?:|$)", s)
    if m:
        return int(m.group(1))

    # Try float threshold 0.5
    try:
        v = float(s)
        return 1 if v >= 0.5 else 0
    except Exception:
        return None


# ----------------------------
# Detector adapters
# ----------------------------

class Detector:
    def predict_file(self, file_path: Path, timeout: float) -> Tuple[Optional[int], str, float]:
        raise NotImplementedError

class CLIDetector(Detector):
    def __init__(self, cmd_template: str):
        """
        cmd_template example: "{exe} {file}"
        You must supply --exe to fill {exe}.
        Allowed placeholders: {exe}, {file}
        """
        self.cmd_template = cmd_template

    def predict_file(self, file_path: Path, timeout: float) -> Tuple[Optional[int], str, float]:
        start = time.time()
        cmd_str = self.cmd_template.format(file=str(file_path), exe="")
        # If user provided the full command in the template, we just split shell-like
        # We'll use shell=True ONLY if template looks complex; otherwise shlex split.
        # For safety and portability, we avoid shell=True; require users to pass simple templates.
        import shlex
        try:
            cmd = shlex.split(cmd_str)
        except Exception:
            cmd = [cmd_str]
        try:
            proc = subprocess.run(cmd, input=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=timeout, check=False)
            elapsed = time.time() - start
            out = (proc.stdout or b"").decode(errors="ignore").strip()
            pred = normalize_prediction(out)
            # Fallback: check last line of stderr
            if pred is None and proc.stderr:
                pred = normalize_prediction(proc.stderr.decode(errors="ignore").strip().splitlines()[-1])
            return pred, out, elapsed
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT", time.time() - start
        except FileNotFoundError as e:
            return None, f"EXEC_NOT_FOUND: {e}", time.time() - start
        except Exception as e:
            return None, f"ERROR: {e}", time.time() - start

class HTTPDetector(Detector):
    def __init__(self, url: str, method: str = "POST", headers: Optional[Dict[str,str]] = None):
        self.url = url
        self.method = method.upper()
        self.headers = headers or {"Content-Type": "application/octet-stream"}

    def predict_file(self, file_path: Path, timeout: float) -> Tuple[Optional[int], str, float]:
        """
        Uses Python's stdlib (urllib) to avoid external deps.
        Expects body text '0' or '1' (or a float).
        """
        import urllib.request
        import urllib.error
        start = time.time()
        data = file_path.read_bytes()
        req = urllib.request.Request(self.url, data=data, method=self.method)
        for k,v in self.headers.items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
                pred = normalize_prediction(body)
                return pred, body.strip(), time.time() - start
        except urllib.error.HTTPError as e:
            return None, f"HTTP_{e.code}", time.time() - start
        except urllib.error.URLError as e:
            return None, f"HTTP_ERROR:{e.reason}", time.time() - start
        except Exception as e:
            return None, f"HTTP_ERROR:{e}", time.time() - start

# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    n_pred_missing = 0
    for r in rows:
        y = r.get("label_true")
        yhat = r.get("label_pred")
        if yhat is None:
            n_pred_missing += 1
            continue
        if y == 1 and yhat == 1:
            tp += 1
        elif y == 0 and yhat == 1:
            fp += 1
        elif y == 0 and yhat == 0:
            tn += 1
        elif y == 1 and yhat == 0:
            fn += 1
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "total_scored": total,
        "missing_predictions": n_pred_missing,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "fpr": fpr, "fnr": fnr,
    }

# ----------------------------
# Main runner
# ----------------------------

def collect_files(input_dir: Path, pattern: str) -> list[Path]:
    if pattern == "**/*":
        return [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        return [p for p in input_dir.rglob(pattern) if p.is_file()]

def run_one(detector: Detector, file_path: Path, timeout: float) -> Dict[str, Any]:
    pred, raw, secs = detector.predict_file(file_path, timeout)
    return {
        "file": str(file_path),
        "sha256": sha256_of_file(file_path),
        "label_pred": pred,
        "model_raw": (raw or "").strip(),
        "latency_sec": round(secs, 4),
        "error": None if (pred is not None) else (raw or "no output"),
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a malware detector over a folder of samples.")
    ap.add_argument("--input-dir", required=True, type=Path, help="Root directory with samples.")
    ap.add_argument("--pattern", default="**/*", help="Glob for files (default: **/*).")
    ap.add_argument("--labels-csv", type=Path, help="Optional CSV with columns path,label.")
    ap.add_argument("--mal-pattern", default="mal|malw", help="Regex for directories that imply malware label=1.")
    ap.add_argument("--good-pattern", default="good|benign|clean", help="Regex for directories that imply goodware label=0.")
    ap.add_argument("--mode", choices=["cli","http"], required=True, help="Detector mode.")
    ap.add_argument("--cmd-template", default="{exe} {file}", help="CLI template (ignored in http mode). Example: './detector {file}'")
    ap.add_argument("--exe", default="", help="Optional executable path for {exe} in cmd-template; you can also bake it into the template.")
    ap.add_argument("--url", help="HTTP endpoint URL for posting file bytes (http mode).")
    ap.add_argument("--http-headers", help='JSON object of headers, e.g. {"Content-Type":"application/octet-stream"}')
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-file timeout seconds.")
    ap.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers.")
    ap.add_argument("--output-csv", type=Path, default=Path("eval_results.csv"), help="Where to save per-file results CSV.")
    ap.add_argument("--dry-run", action="store_true", help="List files and exit without querying detector.")
    args = ap.parse_args(argv)

    files = collect_files(args.input_dir, args.pattern)
    if not files:
        print("No files found. Check --input-dir and --pattern.", file=sys.stderr)
        return 2

    label_map = parse_label_from_csv(args.labels_csv)

    if args.mode == "cli":
        if "{file}" not in args.cmd_template:
            print("In --cmd-template you must include {file}.", file=sys.stderr)
            return 2
        template = args.cmd_template.replace("{exe}", args.exe or "")
        detector: Detector = CLIDetector(template)
    else:
        if not args.url:
            print("--url is required in http mode", file=sys.stderr)
            return 2
        headers = None
        if args.http_headers:
            try:
                headers = json.loads(args.http_headers)
            except json.JSONDecodeError as e:
                print(f"--http-headers is not valid JSON: {e}", file=sys.stderr)
                return 2
        detector = HTTPDetector(args.url, headers=headers)

    if args.dry_run:
        for p in files[:50]:
            print(p)
        if len(files) > 50:
            print(f"... and {len(files) - 50} more")
        return 0

    results = []
    start_all = time.time()

    def _task(p: Path):
        row = run_one(detector, p, args.timeout)
        # set ground truth
        norm_path = os.path.normpath(str(p))
        y = None
        if label_map:
            y = label_map.get(norm_path)
        if y is None:
            y = infer_label(p, args.mal_pattern, args.good_pattern)
        row["label_true"] = y
        return row

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(_task, p): p for p in files}
        done = 0
        last_print = time.time()
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            now = time.time()
            if now - last_print >= 1.5:
                print(f"Progress: {done}/{len(files)} files...", file=sys.stderr)
                last_print = now

    elapsed = time.time() - start_all

    # Write CSV
    fieldnames = ["file","sha256","label_true","label_pred","model_raw","latency_sec","error"]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Metrics (only for rows with known labels)
    rows_with_labels = [r for r in results if r.get("label_true") in (0,1)]
    metrics = compute_metrics(rows_with_labels)

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Files processed      : {len(files)}")
    print(f"Files with labels    : {len(rows_with_labels)}")
    print(f"Total time (s)       : {elapsed:.2f}")
    print(f"Throughput (files/s) : {len(files)/elapsed:.2f}")
    print("\nConfusion Matrix (pred ↓ / true →)")
    print("              |  true=0  |  true=1")
    print("--------------+----------+---------")
    print(f"pred=0        |  {metrics['tn']:8d} |  {metrics['fn']:7d}")
    print(f"pred=1        |  {metrics['fp']:8d} |  {metrics['tp']:7d}")
    print("\nMetrics:")
    print(f"accuracy  : {metrics['accuracy']:.4f}")
    print(f"precision : {metrics['precision']:.4f}")
    print(f"recall    : {metrics['recall']:.4f}")
    print(f"f1        : {metrics['f1']:.4f}")
    print(f"FPR       : {metrics['fpr']:.4f}")
    print(f"TPR       : {1 - metrics['fnr']:.4f}")
    print(f"missing preds (no parse / timeout / error): {metrics['missing_predictions']}")

    print(f"\nPer-file CSV written to: {args.output_csv}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
