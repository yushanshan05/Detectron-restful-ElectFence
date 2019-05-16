dockerfile步骤
功能：此镜像创建的容器会自启一个restful服务，对外的端口为创建容器时映射内部8080端口的外部端口，然后参考/detectron/restful/post.py中的命令发送请求命令。

一、模型下载
下载模型以及配置文件

链接地址 https://pan.baidu.com/s/1bemobrSsswKZhH2_V_FYBw
密码 aibn

下载文件名为：detectron_restful_model.tar

在当前目录运行：sudo tar -xvf detectron_restful_model.tar 
解压文件为：retinanet.yaml  model.pkl 

二、创建镜像
将retinanet.yaml  model.pkl两个文件放在dockerfile同级目录，镜像即可生成
命令：docker build -t imagesname:tag .

三、创建容器
命令：nvidia-docker run -idt -p 9528:8080(端口映射) -v /data:/opt --name detectron_gpu_lg_restful_port9528(容器名) detectron_gpu_restful:v1(镜像名)
注：1、创建容器时用nvidia-docker run
    2、创建容器时不能加 /bin/bash,否则创建容器服务不能自启。
