# Universe snapshot

`universe.json` contains the 503-security S&P 500 snapshot collected for this project on September 18, 2026, from the [Wikipedia constituent table](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies). `Yahoo ticker` normalizes share-class dots to Yahoo's hyphens.

This snapshot is used consistently for downloads and clustering. It is not historical point-in-time index membership. Complete training prices determine eligibility; later returns do not determine the cluster assignment or cointegration selection.

`cache/` is local and ignored by Git. It contains Yahoo Finance adjusted closes and request metadata. Use `python main.py --mode walk-forward --refresh` for a new research download; do not substitute a cache with different ticker/date metadata. Default selected mode uses the original snapshot or the bundled frozen replay inputs in `reproducibility/selected` and rejects revised data.
