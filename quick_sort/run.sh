#!/bin/bash

# 检查是否提供了 size 参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <size>"
    exit 1
fi

SIZE=$1

# 打印 size 参数
echo "Size parameter passed: $SIZE"
# 编译 CUDA 代码

nvcc quick_sort_by_single_thread.cu -o quick_sort_by_single_thread  # 替换为你的文件名
nvcc -rdc=true quick_sort_by_multi_stream.cu -o quick_sort_by_multi_stream  # 替换为你的文件名
nvcc -rdc=true quick_sort_by_dynamic_parallel.cu -o quick_sort_by_dynamic_parallel  # 替换为你的文件名

# 运行程序并传递 size 参数
./quick_sort_by_single_thread $SIZE
./quick_sort_by_multi_stream $SIZE
./quick_sort_by_dynamic_parallel $SIZE