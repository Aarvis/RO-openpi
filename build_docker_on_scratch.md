set -euo pipefail

echo "==> Installing Docker Engine on Ubuntu and moving Docker/containerd storage to /scratch"

# 0) Basic sanity checks
if [ ! -d /scratch ]; then
  echo "ERROR: /scratch does not exist. Aborting."
  exit 1
fi

if ! command -v apt >/dev/null 2>&1; then
  echo "ERROR: This script is for Ubuntu/Debian systems with apt."
  exit 1
fi

# 1) Remove conflicting old packages if present
sudo apt-get update
sudo apt-get remove -y docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc || true

# 2) Install prerequisites
sudo apt-get install -y ca-certificates curl

# 3) Add Docker's official GPG key and apt repository
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# 4) Install Docker Engine + Buildx + Compose plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5) Create scratch-backed storage locations
sudo mkdir -p /scratch/docker
sudo mkdir -p /scratch/containerd
sudo chmod 755 "/scratch/docker" "/scratch/containerd"

# 6) Configure Docker daemon to use /scratch/docker
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<'EOF'
{
  "data-root": "/scratch/docker"
}
EOF

# 7) Configure containerd root to use /scratch/containerd
#    This matters on fresh Docker Engine 29+ installs where image/snapshot data may live in /var/lib/containerd.
if [ ! -f /etc/containerd/config.toml ]; then
  containerd config default | sudo tee /etc/containerd/config.toml > /dev/null
fi

sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak.$(date +%Y%m%d_%H%M%S)
sudo sed -i 's#^\s*root = ".*"#root = "/scratch/containerd"#' /etc/containerd/config.toml

# 8) Enable and restart services
sudo systemctl enable containerd docker
sudo systemctl restart containerd
sudo systemctl restart docker

# 9) Verify installation
echo
echo "==> Versions"
sudo docker --version
sudo docker compose version
sudo docker buildx version

echo
echo "==> Service status"
sudo systemctl --no-pager --full status docker | sed -n '1,15p'
echo
sudo systemctl --no-pager --full status containerd | sed -n '1,15p'

echo
echo "==> Storage locations"
sudo docker info --format 'Docker Root Dir: {{.DockerRootDir}}'
grep -E '^\s*root = ' /etc/containerd/config.toml || true

echo
echo "==> Running hello-world test"
sudo docker run --rm hello-world

echo
echo "==> Done"
echo "Docker data:      /scratch/docker"
echo "containerd data:  /scratch/containerd"
echo
echo "Use sudo docker ... for commands."
echo "Example test:"
echo "  sudo docker pull ubuntu:22.04"
echo "  sudo docker run --rm -it ubuntu:22.04 bash"