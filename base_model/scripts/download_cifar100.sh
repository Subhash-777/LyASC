#!/usr/bin/env bash
# Downloads and verifies CIFAR-100, per plan.md Section 3.
# If cs.toronto.edu is blocked from this machine's network, download the file
# manually via a browser from https://www.cs.toronto.edu/~kriz/cifar.html and
# place it at data/cifar100/cifar-100-python.tar.gz, then re-run this script —
# it will skip the download step and just verify + extract.
set -euo pipefail

TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/cifar100"
URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
EXPECTED_MD5="eb9058c3a382ffc7106e4002c42a8d85"
ARCHIVE="${TARGET_DIR}/cifar-100-python.tar.gz"

mkdir -p "${TARGET_DIR}"

if [ ! -f "${ARCHIVE}" ]; then
  echo "Downloading CIFAR-100 to ${ARCHIVE} ..."
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
echo "Done. Expect ${TARGET_DIR}/cifar-100-python/ to contain train, test, meta"
