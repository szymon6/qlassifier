#!/bin/bash
# Fetch the stage-2 raw inputs into experiments/data/{downloads,raw} (gitignored) and verify their SHA-256.
# Up: ../data/README.md (sources, licences, hashes) · run from the repo root: bash experiments/hardware/common/fetch_data.sh
set -eu
D=experiments/hardware/data
mkdir -p $D/downloads/spambase $D/downloads/banknote $D/downloads/haberman $D/raw/uci_beijing $D/raw/uscrn
get() { [ -f "$2" ] || curl -sSL -o "$2" "$1"; }
get https://archive.ics.uci.edu/static/public/94/spambase.zip $D/downloads/spambase/spambase.zip
get "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip" $D/downloads/banknote/banknote.zip
get "https://archive.ics.uci.edu/static/public/43/haberman+s+survival.zip" $D/downloads/haberman/haberman.zip
get https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip $D/raw/uci_beijing/beijing.zip
for y in 2021 2022 2023; do
  get https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/$y/CRNH0203-$y-CO_Boulder_14_W.txt $D/raw/uscrn/CO_Boulder_14_W_$y.txt
done
get https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/headers.txt $D/raw/uscrn/headers.txt
for d in spambase banknote haberman; do (cd $D/downloads/$d && unzip -o -q *.zip); done
(cd $D/raw/uci_beijing && unzip -o -q beijing.zip && unzip -o -q PRSA2017_Data_20130301-20170228.zip 2>/dev/null || true
 f=$(find . -name 'PRSA_Data_Dongsi_*.csv' | head -1); [ -n "$f" ] && cp "$f" . || true)
shasum -a 256 -c - <<'SUMS'
b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12  experiments/hardware/data/downloads/spambase/spambase.data
d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9  experiments/hardware/data/downloads/banknote/data_banknote_authentication.txt
b4b7a32586a5668f9f4d6dc8be9d1bc8cd4822523affb1f6b5bfc350681ef3e2  experiments/hardware/data/downloads/haberman/haberman.data
9c0020214425e5ee2a64d42f97a1f632e1d8f6b53ad50edd85a686cc7999f27e  experiments/hardware/data/raw/uci_beijing/PRSA_Data_Dongsi_20130301-20170228.csv
b1c389b5f45c7312d243b06cc1f8cca5dd971687d6a0c79defee9f1a1ed9aa11  experiments/hardware/data/raw/uscrn/CO_Boulder_14_W_2021.txt
e2c8b22fcef736cc3a287bde5a3267af4be2985026cd1802ecd65b5a9f6e9cc5  experiments/hardware/data/raw/uscrn/CO_Boulder_14_W_2022.txt
db472400baedd545dfab49ed7db33b7da78deb9cc29faaa5821afc51e98b896b  experiments/hardware/data/raw/uscrn/CO_Boulder_14_W_2023.txt
44564d2ca37fe7c2e893495c7fd339621817052e70373c3bd562d6fbdd1375a4  experiments/hardware/data/raw/uscrn/headers.txt
SUMS
