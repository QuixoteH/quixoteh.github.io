# Awards archive

Archived copies of the official leaderboard pages behind the results listed on the homepage, kept in case the organizers' sites go offline. This folder is not part of the published site (GitHub Pages only deploys `out/`).

Captured on 2026-11-08 with headless Chrome. Every screenshot is the web page only (no browser UI) and includes the page header and title together with the leaderboard.

| Competition | Result | Row in the capture | Folder |
|---|---|---|---|
| NVIDIA Video-to-Data (V2D) Challenge @ CoRL 2026, Track 1 | 34th | `34 quixoteh`, CD-H 14.9138, CD-O 67.8452 | [`v2d-challenge-corl-2026/`](v2d-challenge-corl-2026/) |
| RoboSynChallenge @ NeurIPS 2026, preliminary stage | 23rd | `23 LBWSquare`, score 34.0, success 40.1% | [`robosynchallenge-neurips-2026/`](robosynchallenge-neurips-2026/) |

## Files in each folder

| File | Content |
|---|---|
| `leaderboard-view.png` | Page header, title and the whole leaderboard table in one screenshot |
| `full-page.png` | Entire page as one image (V2D only; the RoboSyn page is too tall for one image) |
| `full-page.pdf` | Entire page printed to PDF |
| `page.mhtml` | Self-contained snapshot of the page, openable in Chrome or Edge |
| `capture-info.json` | Source URL, capture time (UTC), the row for this entry, and SHA-256 of every file |

## Notes

- V2D: the Track 1 tab is sorted by CD-H, and the rank column follows that order. The page states "Updated Thu, 05 Nov 2026 01:43:17 GMT" under the table, after the leaderboard freeze on 2026-11-04 17:00 EST.
- RoboSyn: the page's "Average" view, 47 results.
- Sources: <https://nvidia-isaac.github.io/video_to_data/v2d_challenge/> and <https://robosyn-bench.net/#/leaderboard>.
