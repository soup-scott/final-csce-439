#!/usr/bin/env python3
import os
import requests
import argparse

API_URL = "http://127.0.0.1:8080/"

def send_file(path, label):
    with open(path, "rb") as f:
        data = f.read()

    try:
        resp = requests.post(
            API_URL,
            params={"label": label},
            data=data,
            headers={"Content-Type": "application/octet-stream"},
            # timeout=5
        )
        return resp.status_code, resp.text
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Directory containing samples")
    parser.add_argument("--label", required=True, choices=["0", "1"])
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.recursive:
        files = [
            os.path.join(root, name)
            for root, dirs, names in os.walk(args.directory)
            for name in names
        ]
    else:
        files = [
            os.path.join(args.directory, name)
            for name in os.listdir(args.directory)
            if os.path.isfile(os.path.join(args.directory, name))
        ]

    print(f"Found {len(files)} files to send.\n")

    for path in files:
        status, msg = send_file(path, args.label)
        print(f"[{status}] {path}")
        if status != 200:
            print("  -> ERROR:", msg)


if __name__ == "__main__":
    main()
