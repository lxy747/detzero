# run docker container  
镜像文件：preanno_saved_detzero_v2.tar  
```shell
sudo docker load -i preanno_saved_detzero_v2.tar

sudo docker run -it \
    -v /local_code_dir:/code \ # code dir
    -v /mnt:/mnt  \  # test data dir
    --network host \
    --privileged \ # 提升权限，方便操作网卡/设备
    --gpus all --name your_container_name preanno:v2 /bin/bash
```
local_code_dir could be like this:
```
local_code_dir
      └── OD
          ├── ckpts   # weights
          ├── detzero_dev  # 工程目录， proj_dir
          ├── readme_deploy.md
          └── scripts  # 测试脚本
              ├── premarking_pipelines_t68.sh # 测试脚本
              └── test_vis_results_t68.sh
```

Attention:  
* the host machine must have installed cuda. see [docker.md](./docker.md)  

* If it does not work well, you could compile detzero.
```shell
  # proj_dir is the project dir, e.g. detzero_dev
  cd ${proj_dir}/utils && python setup.py develop
  cd ${proj_dir}/detection && python setup.py develop
  cd ${proj_dir}/tracking && python setup.py develop
```
# demo  
执行脚本  
```shell
bash scripts/premarking_pipelines_t68.sh
```
测试数据输入参数在scripts/premarking_pipelines_t68.sh里面配置。  
* 参数1：测试数据的根目录  
  > data_dir=/mnt/cfs/e2e/datasets/t68_260402  
* 参数2：测试clip制作成的pkl文件, 相对于data_dir的目录    
  > test_pkl=pkls_motovis_demotion/toXM/20260122-15-13-10.pkl  
* 参数3：输出目录，自定  
  > output_dir=your_output_dir    
