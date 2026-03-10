set -euo pipefail

DISK="/dev/nvme0n1"
PART="/dev/nvme0n1p1"
MOUNT_POINT="/scratch"
USER_NAME="snorbyte_admin"

echo "==> This will ERASE everything on $DISK"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINT,UUID "$DISK"

echo "==> Unmounting anything stale"
sudo umount "$PART" 2>/dev/null || true
sudo umount "$MOUNT_POINT" 2>/dev/null || true

echo "==> Wiping old signatures"
sudo wipefs -a "$DISK" || true

echo "==> Creating GPT partition table"
sudo parted -s "$DISK" mklabel gpt

echo "==> Creating one full-size partition"
sudo parted -s -a optimal "$DISK" mkpart primary ext4 1MiB 100%

echo "==> Refreshing partition table"
sudo partprobe "$DISK"
sudo udevadm settle

echo "==> Waiting for partition node"
for i in $(seq 1 20); do
  [ -b "$PART" ] && break
  sleep 1
done
[ -b "$PART" ]

echo "==> Formatting partition as ext4"
sudo mkfs.ext4 -F -L scratch "$PART"

echo "==> Creating mount point"
sudo mkdir -p "$MOUNT_POINT"

echo "==> Getting UUID"
UUID="$(sudo blkid -s UUID -o value "$PART")"
echo "UUID=$UUID"

echo "==> Backing up /etc/fstab"
sudo cp /etc/fstab /etc/fstab.backup.$(date +%F-%H%M%S)

echo "==> Removing old /scratch entries from /etc/fstab"
sudo awk '$2 != "/scratch"' /etc/fstab | sudo tee /etc/fstab >/dev/null

echo "==> Adding new /scratch entry"
echo "UUID=$UUID $MOUNT_POINT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

echo "==> Mounting"
sudo mount "$PART" "$MOUNT_POINT"

echo "==> Creating folders"
sudo mkdir -p "$MOUNT_POINT/hf" "$MOUNT_POINT/tmp"

echo "==> Setting ownership and permissions"
sudo chown -R "$USER_NAME:$USER_NAME" "$MOUNT_POINT"
sudo chmod 755 "$MOUNT_POINT" "$MOUNT_POINT/hf" "$MOUNT_POINT/tmp"

echo "==> Verification"
findmnt "$MOUNT_POINT"
df -h "$MOUNT_POINT"
ls -ld "$MOUNT_POINT" "$MOUNT_POINT/hf" "$MOUNT_POINT/tmp"
lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINT,UUID