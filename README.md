## CMakePresets.json 中的问题
  在ICT2026-PdistPdist/CMakePresets.json文件中，第42行处：
  

  
                  "ASCEND_CANN_PACKAGE_PATH": {
                  
                    "type": "PATH",
                    
                    "value": "/home/shouss/miniconda3/envs/cann/Ascend/ascend-toolkit/latest" 
                    
                },
这里的value路径值貌似是创建算子时自动生成的，因此一开始的路径并不是我自己的，编译时会出现问题。

然后我尝试了用环境变量${ASCEND_HOME_PATH}赋值，照理说这里的cmake版本是支持的，并且下面的配置文件里也使用了，但是仍然有问题，cmake无法解析这里的相对路径。因此我们协作的时候可能会出现一些小麻烦，需要手动更改为自己路径，后面再试试别的宏定义，先就这样吧

## 核函数
  核函数侧的实现不能用我们熟知的语言来编写，会出现各种各样的问题，包括API无法调用，不能直接定义值等等，而且高度依赖AscendC中各类没见过的但似曾相识的函数。
  因此这里只是列出了一个框架，实际计算部分的实现还需要学习以下AscendC的语法

## 编译和算子安装
  bash.sh运行后会在buildout里生成.run文件，运行.run --install-path会默认将算子安装进入环境里的ascend-toolkit的算子库里。运行命令后提示会有路径的。
