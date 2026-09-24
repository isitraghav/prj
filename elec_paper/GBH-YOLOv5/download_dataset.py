"""Download the PV-Multi-Defect dataset from its public GitHub portal.

Portal: https://github.com/CCNUZFW/PV-Multi-Defect (Annotations/ + JPEGImages/).
The dataset is not committed to this repo; fetch it here before data prep.

    uv run python download_dataset.py
    uv run python prepare_pv_data.py   # defaults to ./PV-Multi-Defect

Stdlib only (urllib + zipfile), so it runs before any deps are installed.
"""
import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://codeload.github.com/CCNUZFW/PV-Multi-Defect/zip/refs/heads/main"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="PV-Multi-Defect", help="target dataset dir")
    ap.add_argument("--url", default=ZIP_URL)
    ap.add_argument("--force", action="store_true", help="re-download over existing")
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "Annotations").is_dir() and (out / "JPEGImages").is_dir() and not args.force:
        print(f"already present: {out.resolve()}  (use --force to re-download)")
        return

    print(f"downloading {args.url} ...")
    with urllib.request.urlopen(args.url) as r:  # noqa: S310 (trusted GitHub URL)
        data = r.read()
    print(f"got {len(data) / 1e6:.1f} MB, extracting ...")

    tmp = out.parent / f"{out.name}__tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(tmp)

    # GitHub zips wrap everything in <repo>-<branch>/ — find the folder that
    # actually holds Annotations/ and move it into place.
    roots = [p for p in tmp.rglob("Annotations") if p.is_dir()]
    assert roots, "Annotations/ not found in downloaded archive"
    src = roots[0].parent
    if out.exists():
        shutil.rmtree(out)
    shutil.move(str(src), str(out))
    shutil.rmtree(tmp, ignore_errors=True)

    n_ann = len(list((out / "Annotations").glob("*.xml")))
    n_img = len(list((out / "JPEGImages").glob("*.jpg")))
    print(f"done: {out.resolve()}  annotations={n_ann}  images={n_img}")


if __name__ == "__main__":
    main()
