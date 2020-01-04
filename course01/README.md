# TA_class01

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matteosoo/course01/blob/master/TA_class01.ipynb)

## ML and DL - Warm Up
- Python Installation
    - Anaconda: https://www.anaconda.com/distribution/
    - Test Anaconda Environment
        - ![](https://i.imgur.com/4ny4kQk.png)
        - ![](https://i.imgur.com/Tj4s03E.png)
    - Miniconda: https://docs.conda.io/en/latest/miniconda.html
    - Command on python
        - Go to Terminal (Mac/Linux) or Anaconda Prompt (Windows)
        - Install Package
            - pip install {package name}
            - conda install {package name}
        - heck how many packages you have
            - pip freeze
    - Virtual Environment
        - Develop projects
            - Manage various needs effectively 
            - Avoid packages version collision
- Implement on GUI: 
    - ![](https://i.imgur.com/xJqwUXH.png)

    - ![](https://i.imgur.com/8Ubaqqm.png)

- Deep Learning Environment
    - Note: You should have your own GPU, and recommend you for using Ubuntu OS.
    - Download **Cuda**: https://developer.nvidia.com/cuda-downloads
        - ![](https://i.imgur.com/aOnstw4.png)
    - Download **Cudnn**: https://developer.nvidia.com/cudnn
        - Register an account
            - ![](https://i.imgur.com/LmJhoxL.png)
        - Go to E-mail to activate your account
            - ![](https://i.imgur.com/G6cNYky.png)
        - Download cudnn for your CUDA version
            - ![](https://i.imgur.com/Un5r338.png)
        - Cudnn is a compressed file.
            - Unzip the Cudnn file
            - Find the Cuda file (previously downloaded)
            - cd cuda
            - Put Cudnn unzipped file into Cuda file
                - ![](https://i.imgur.com/O3tQHDV.png)
    - There are a lot of deep learning packages, such as Theano*, Caffe, TensorFlow, PyTorch, Keras, and so on.
        - ![](https://i.imgur.com/sgwD1K9.png)
    - Deep Learning Package - Pytorch: https://pytorch.org/
        - ![](https://i.imgur.com/uy0aBq7.png)
        - Test Pytorch 
            - Go to your Power Shell / cmd / Terminal
            - ![](https://i.imgur.com/OnOy6WC.png)
            - ![](https://i.imgur.com/1Uukfht.png)
            - If output is **true**, it represents your GPU is loaded 
            - \>>> quit() //you will end of the python program


###### tags: `MachineLearing` `DeepLearing`