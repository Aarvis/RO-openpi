#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG
# =========================
USER_NAME="snorbyte_admin"
MOUNT_POINT="/scratch"

# Prefer Azure's stable local-disk path if present; otherwise fall back to nvme0n1
DISK_LINK="/dev/disk/azure/local/by-index/0"
DISK_FALLBACK="/dev/nvme0n1"

# ext4 is fine here; Azure examples often use XFS, but ext4 works too for scratch
FS_TYPE="ext4"
FS_LABEL="scratch"

# =========================
# RESOLVE DISK
# =========================
if [ -e "$DISK_LINK" ]; then
  DISK="$(readlink -f "$DISK_LINK")"
else
  DISK="$DISK_FALLBACK"
fi

PART="${DISK}p1"
if [[ "$DISK" =~ ^/dev/nvme ]]; then
  PART="${DISK}p1"
else
  PART="${DISK}1"
fi

echo "==> Using disk: $DISK"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINT,UUID "$DISK" || true

# =========================
# PREP MOUNT POINT
# =========================
sudo mkdir -p "$MOUNT_POINT"

# If already mounted, just ensure folders/permissions and exit cleanly
if mountpoint -q "$MOUNT_POINT"; then
  echo "==> $MOUNT_POINT already mounted"
else
  # If partition exists and has a filesystem, mount it
  if [ -b "$PART" ] && sudo blkid "$PART" >/dev/null 2>&1; then
    echo "==> Found existing filesystem on $PART"
    sudo mount "$PART" "$MOUNT_POINT"
  else
    echo "==> No usable filesystem found; initializing scratch disk"
    sudo umount "$PART" 2>/dev/null || true
    sudo umount "$MOUNT_POINT" 2>/dev/null || true

    # Wipe only the scratch disk
    sudo wipefs -a "$DISK" || true

    # Partition
    sudo parted -s "$DISK" mklabel gpt
    sudo parted -s -a optimal "$DISK" mkpart primary "$FS_TYPE" 1MiB 100%

    sudo partprobe "$DISK"
    sudo udevadm settle

    for _ in $(seq 1 20); do
      [ -b "$PART" ] && break
      sleep 1
    done
    [ -b "$PART" ]

    # Format
    if [ "$FS_TYPE" = "ext4" ]; then
      sudo mkfs.ext4 -F -L "$FS_LABEL" "$PART"
    elif [ "$FS_TYPE" = "xfs" ]; then
      sudo mkfs.xfs -f -L "$FS_LABEL" "$PART"
    else
      echo "Unsupported FS_TYPE=$FS_TYPE"
      exit 1
    fi

    sudo mount "$PART" "$MOUNT_POINT"
  fi
fi

# =========================
# FOLDERS + PERMS
# =========================
sudo mkdir -p "$MOUNT_POINT/hf/datasets" "$MOUNT_POINT/tmp"
sudo chown -R "$USER_NAME:$USER_NAME" "$MOUNT_POINT"
sudo chmod 755 "$MOUNT_POINT" "$MOUNT_POINT/hf" "$MOUNT_POINT/hf/datasets" "$MOUNT_POINT/tmp"

echo "==> Done"
findmnt "$MOUNT_POINT"
df -h "$MOUNT_POINT"
ls -ld "$MOUNT_POINT" "$MOUNT_POINT/hf" "$MOUNT_POINT/hf/datasets" "$MOUNT_POINT/tmp"