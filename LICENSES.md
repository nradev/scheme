# Third-Party Licenses

SCHEME's own code is under the MIT License (see [`LICENSE`](LICENSE)). This repository also
contains third-party code and data, each covered by its own license, as listed below.

## Vendored libraries

| Component | Version | Location | License | Upstream |
|---|---|---|---|---|
| lightkurve | 2.5.2dev | [`scheme/settings/exoplanet_detection/setting_source/lightkurve/`](scheme/settings/exoplanet_detection/setting_source/lightkurve/) | MIT ([`LICENSE`](scheme/settings/exoplanet_detection/setting_source/lightkurve/LICENSE)) | https://github.com/lightkurve/lightkurve |
| skfolio | 0.15.3 | [`scheme/settings/portfolio_optimization/setting_source/skfolio/`](scheme/settings/portfolio_optimization/setting_source/skfolio/) | BSD 3-Clause ([`LICENSE`](scheme/settings/portfolio_optimization/setting_source/skfolio/LICENSE)) | https://github.com/skfolio/skfolio |

The full license text for each library is in the `LICENSE` file in its directory and must be kept
with any copy of that code.

The vendored copies are included in full. Each setting's `patches.py` modifies them when the
evaluation sandbox is built, to create the benchmark tasks. These modifications are not
endorsed by the original authors.

## Data

| File | Description | Source |
|---|---|---|
| [`scheme/settings/exoplanet_detection/setting_source/data/lightcurve.npz`](scheme/settings/exoplanet_detection/setting_source/data/lightcurve.npz) | HAT-P-7b light curve with added noise | NASA Kepler mission |
| `skfolio/src/skfolio/datasets/data/*.csv.gz` | S&P 500 and factor datasets | Bundled with skfolio, under skfolio's license |
