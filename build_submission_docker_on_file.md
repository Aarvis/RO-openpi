sudo cp /etc/containerd/config.toml /etc/containerd/config.toml.bak.$(date +%Y%m%d_%H%M%S)

sudo sh -c 'printf "root = \"/scratch/containerd\"\nstate = \"/run/containerd\"\n\n%s" "$(cat /etc/containerd/config.toml)" > /etc/containerd/config.toml'

sudo mkdir -p /scratch/containerd
sudo chmod 755 /scratch/containerd

sudo systemctl restart containerd
sudo systemctl restart docker

sudo grep -n '^root = ' /etc/containerd/config.toml
sudo grep -n '^state = ' /etc/containerd/config.toml



docker buildx rm scratchbuilder 2>/dev/null || true
docker buildx create --name scratchbuilder --driver docker-container --use
docker buildx inspect --bootstrap
docker buildx ls

docker buildx build --load -t lehome-openpi-submission -f scripts/docker/lehome_submission.Dockerfile .
docker run --rm --gpus all -p 8080:8080 lehome-openpi-submission

aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 471183654236.dkr.ecr.ap-south-1.amazonaws.com

docker buildx build \
  --push \
  -t 471183654236.dkr.ecr.ap-south-1.amazonaws.com/dummy_checkpoint_40k:latest \
  -f scripts/docker/lehome_submission.Dockerfile .

docker pull 471183654236.dkr.ecr.ap-south-1.amazonaws.com/dummy_checkpoint_40k:latest

docker run --rm --gpus all -p 8000:8080 \
  471183654236.dkr.ecr.ap-south-1.amazonaws.com/dummy_checkpoint_40k:latest


docker buildx build --load -t lehome-openpi-submission -f scripts/docker/lehome_submission.Dockerfile .

docker tag lehome-openpi-submission aarvisarchive/dume_ft_with_state_10e:latest

docker push aarvisarchive/dume_ft_with_state_10e:latest



docker buildx build --load -t lehome-openpi-submission-b_4_ft_20 -f scripts/docker/lehome_submission.Dockerfile .





docker tag lehome-openpi-submission-b_4_ft_20 aarvisarchive/dume_b_4_ft_20:latest

docker push aarvisarchive/dume_b_4_ft_20:latest


docker run --rm --gpus all -p 8000:8080 aarvisarchive/dume_b_4_ft_10:latest


docker buildx build --load -t lehome-openpi-submission-rl_b_4_ft_10 -f scripts/docker/lehome_submission_rl_critic.Dockerfile .


docker tag lehome-openpi-submission-rl_b_4_ft_10 aarvisarchive/dume_rl_b_4_ft_10:latest

docker tag lehome-openpi-submission-rl_b_4_ft_20 aarvisarchive/dume_rl_b_4_ft_20:latest

docker push aarvisarchive/dume_rl_b_4_ft_10:latest

docker push aarvisarchive/dume_rl_b_4_ft_20:latest

aarvisarchive/dume_rl_b_4_ft_10