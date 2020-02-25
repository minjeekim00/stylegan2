nvidia-docker run -it \
	-v /mnt/nas100/minjee.kim:/root/projects \
	-v /mnt/mini_nas:/root/mini_nas \
	-v /mnt/scratch4TB/minjee:/root/datasets \
        -p 2222-2224:2222-2224  \
        m40030811/stylegan2:latest /bin/bash

export containerId=$(docker ps -l -q)

xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`
docker start $containerId
