#!/usr/bin/env bash
# Downloads and verifies CIFAR-10, per plan.md Section 3.
# If cs.toronto.edu is blocked from this machine's network, download the file
# manually via a browser from https://www.cs.toronto.edu/~kriz/cifar.html and
# place it at data/cifar10/cifar-10-python.tar.gz, then re-run this script —
# it will skip the download step and just verify + extract.
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/cifar10"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
EXPECTED_MD5="c58f30108f718f92721af3b95e74349"
ARCHIVE="${TARGET_DIR}/cifar-10-python.tar.gz"

mkdir -p "${TARGET_DIR}"

if [ ! -f "${ARCHIVE}" ]; then
  echo "Downloading CIFAR-10 to ${ARCHIVE} ..."
  curl -L -o "${ARCHIVE}" "${URL}"
else
  echo "Archive already present at ${ARCHIVE}, skipping download."
fi

echo "Verifying checksum..."
ACTUAL_MD5="$(md5sum "${ARCHIVE}" | awk '{print $1}')"
if [ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]; then
  echo "ERROR: MD5 mismatch. Expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}." >&2
  echo "Re-download manually from ${URL} and retry." >&2
  exit 1
fi
echo "Checksum OK."

echo "Extracting..."
tar -xzf "${ARCHIVE}" -C "${TARGET_DIR}"
echo "Done. Expect ${TARGET_DIR}/cifar-10-batches-py/ to contain data_batch_1..5, test_batch, batches.meta"
