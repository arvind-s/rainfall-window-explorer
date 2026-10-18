#!/usr/bin/env python3
"""WeatherNext 3 15-day rainfall forecast for every block (polygon mean, daily IST totals).

Uses the partial-chunk ("prefix") read from ~/Downloads/python_prefix.py: each lead's global
zstd chunk is decoded only up to the northern-most block's row, and every block's cells are
read from that one buffer, so 320 blocks cost the same as one point. Each lead's cell values
are cached in data/wn3_cache/<run>/ so an interrupted run resumes.

    gcloud auth application-default login      # allowlisted account
    python scripts/wn3_forecast.py              # newest complete run, 360 leads
"""
import argparse
import datetime as dt
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import zstandard
from shapely.geometry import Point

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/Downloads"))
from python_prefix import N_LAT, N_LON, RES, Bucket, check_layout, latest_run  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IST = dt.timedelta(hours=5, minutes=30)


def block_cells(gdf):
    """{block_id: flat cell indices} -- 0.1° cell centres inside the polygon, else nearest cell."""
    out = {}
    for _, r in gdf.iterrows():
        g = r.geometry
        x0, y0, x1, y1 = g.bounds
        rows = range(int(np.floor((y0 + 90) / RES)), int(np.ceil((y1 + 90) / RES)) + 1)
        cols = range(int(np.floor(x0 / RES)), int(np.ceil(x1 / RES)) + 1)
        cells = [ri * N_LON + ci for ri in rows for ci in cols
                 if g.contains(Point(ci * RES, -90 + ri * RES))]
        if not cells:
            c = g.representative_point()
            cells = [round((c.y + 90) / RES) * N_LON + round(c.x / RES) % N_LON]
        out[r.block_id] = np.array(cells)
    return out


def read_cells(b, key, flat, ratio):
    """Values at flat float32 indices from one zstd chunk, decoding only the needed prefix."""
    need = (int(flat.max()) + 1) * 4
    dec = zstandard.ZstdDecompressor().decompressobj()
    pos, size, buf = 0, int(need * ratio) + (512 << 10), bytearray()
    while len(buf) < need:
        data = b.get(key, pos, pos + size - 1)
        if not data:
            raise RuntimeError(f"{key}: stream ended at {len(buf)} of {need} bytes")
        pos += len(data)
        buf += dec.decompress(data)
        size = max(size // 4, 1 << 20)  # top-ups are small; first guess covers most of it
    return np.frombuffer(bytes(buf[:need]), dtype="<f4")[flat]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default="total_precipitation_1hr_mean")
    ap.add_argument("--leads", type=int, default=360)
    ap.add_argument("--run", help="run folder (default: newest complete)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--ratio", type=float, default=0.95)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "out"))
    args = ap.parse_args()

    b = Bucket(args.concurrency)
    if args.run:
        run = args.run
        init = dt.datetime.strptime(run[:11], "%Y%m%d_%H").replace(tzinfo=dt.timezone.utc)
    else:
        run, init = latest_run(b, args.leads, args.array)
    check_layout(b, run, args.array, args.leads)
    print(f"run {run} ({args.array}, {args.leads} leads)", flush=True)

    gdf = gpd.read_file(os.path.join(args.out, "blocks.geojson"))
    cells = block_cells(gdf)
    flat = np.unique(np.concatenate(list(cells.values())))
    print(f"{len(cells)} blocks -> {len(flat)} cells, decode {flat.max() * 4 / 2**20:.1f} of "
          f"{N_LAT * N_LON * 4 / 2**20:.1f} MiB per lead", flush=True)

    cache = os.path.join(HERE, "data", "wn3_cache", run, args.array)
    os.makedirs(cache, exist_ok=True)

    def lead(i):
        f = os.path.join(cache, f"{i:03d}.npy")
        if not os.path.exists(f):
            for attempt in range(5):  # GCS reads occasionally time out on a slow link
                try:
                    v = read_cells(b, f"{run}/predictions.zarr/{args.array}/c/{i}/0/0", flat, args.ratio)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 4:
                        raise
                    print(f"  lead {i}: {type(e).__name__}, retry {attempt + 1}", flush=True)
                    time.sleep(5 * (attempt + 1))
            np.save(f + ".tmp.npy", v)
            os.replace(f + ".tmp.npy", f)
        return i, np.load(f)

    vals, t0, done = {}, time.time(), 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        for fut in as_completed([pool.submit(lead, i) for i in range(args.leads)]):
            i, v = fut.result()
            vals[i] = v
            done += 1
            if done % 20 == 0 or done == args.leads:
                print(f"  {done}/{args.leads} leads  {time.time() - t0:.0f}s  "
                      f"{b.bytes / 2**30:.2f} GiB", flush=True)

    # hourly grid values (m) -> block polygon means (mm), valid hour = end of the 1-h accumulation
    pos = {c: k for k, c in enumerate(flat)}
    idx = {bid: [pos[c] for c in cs] for bid, cs in cells.items()}
    rows = []
    for i in range(args.leads):
        end_ist = (init + dt.timedelta(hours=i + 1) + IST).replace(tzinfo=None)
        day = (end_ist - dt.timedelta(minutes=1)).date()  # accumulation mostly in this IST day
        v = np.clip(vals[i], 0, None) * 1000.0
        for bid, k in idx.items():
            rows.append((bid, day, float(v[k].mean())))
    hourly = pd.DataFrame(rows, columns=["block_id", "date", "rain_mm"])
    daily = hourly.groupby(["block_id", "date"], as_index=False).agg(
        rain_mm=("rain_mm", "sum"), hours=("rain_mm", "size"))
    meta = gdf[["block_id", "block", "district", "state"]]
    daily = meta.merge(daily, on="block_id")
    daily["run_utc"] = init.strftime("%Y-%m-%d %H:%M")
    daily["date"] = pd.to_datetime(daily["date"])

    p = os.path.join(args.out, "forecast_wn3.parquet")
    daily.to_parquet(p, index=False)
    wide = daily.pivot_table(index=["block_id", "block", "district", "state"], columns="date",
                             values="rain_mm").round(1)
    wide.columns = [c.strftime("%Y-%m-%d") for c in wide.columns]
    wide["total_15d_mm"] = wide.sum(axis=1).round(1)
    csv = os.path.join(HERE, f"wn3_forecast_{init:%Y%m%d_%H}z.csv")
    wide.reset_index().to_csv(csv, index=False)
    print(f"wrote {p} ({len(daily)} rows) and {csv}", flush=True)


if __name__ == "__main__":
    main()
