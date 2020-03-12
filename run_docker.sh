sudo docker run -it \
        --gpus all \
        --name minjee_stylegan2 \
        --workdir /root/projects \
	-v /mnt/nas100/minjee.kim:/root/projects \
	-v /mnt/mini_nas:/root/mini_nas \
	-v /mnt/scratch4TB/minjee:/root/datasets \
        -p 2222-2224:2222-2224  \
	--shm-size 48G \
        -c 512 \
        --restart always \
        stylegan2:default 
#m40030811/stylegan2:latest /bin/bash
#export containerId=$(docker ps -l -q)
#xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`
#sudo docker start minjee_stylegan2

