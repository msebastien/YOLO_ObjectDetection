#!/usr/bin/env bash
# Written by Sébastien Maes

IS_VENV_INITIALIZED=false
PROJECT_DIRECTORY_NAME=YOLO_ObjectDetection
PROJECT_PATH=$(find "$HOME" -type d -name $PROJECT_DIRECTORY_NAME)

# Checks if there is a python virtual environment and activate it
check_python_venv () {
    if [ -d "$1"/.venv ]; then
        echo -n "A Python virtual environment has already been created. Activate it."
    else
        echo -n "No Python virtual environment. A new one will be created."
        python3 -m venv .venv --system-site-packages
    fi

    if [[ "$IS_VENV_INITIALIZED" == false ]]; then
        source "$1"/.venv/bin/activate
        IS_VENV_INITIALIZED=true
    fi
}

install_pypi_packages () {
    cd "$PROJECT_PATH" || exit
    check_python_venv "$(pwd)"

    python3 -m pip install -U -v    \
    python-magic                    \
    numpy                           \
    pysdl2                          \
    pysdl2-dll                      \
    huggingface-hub                 \
    supervision

    echo -n "========================================================================"
    echo -n "|   The PyPI packages install script has completed its execution.      |"
    echo -n "========================================================================"
}

install_opencv_cuda() {
    cd "$HOME"

    # Clone opencv-python source repository, including git submodules
    git clone --recursive https://github.com/opencv/opencv-python.git
    cd opencv-python

    # Export CMake Build Flags
    # Build Flags: https://docs.opencv.org/4.11.0/db/d05/tutorial_config_reference.html

    # NVIDIA CUDA Support
    export CMAKE_ARGS="-D WITH_CUDA=ON -D WITH_CUDNN=ON -D WITH_CUFFT=ON -D WITH_CUBLAS=ON -D WITH_NVCUVID=ON"
    # Enable CPU Optimizations by using AVX2 instructions
    export CMAKE_ARGS="${CMAKE_ARGS} -D CPU_BASELINE=AVX2"
    # Enable Build Hardening
    export CMAKE_ARGS="${CMAKE_ARGS} -D ENABLE_BUILD_HARDENING=ON"
    # Enable Link Time Optimization (LTO)
    export CMAKE_ARGS="${CMAKE_ARGS} -D ENABLE_LTO=ON"
    # Include OpenCV Examples ?
    export CMAKE_ARGS="${CMAKE_ARGS} -D BUILD_EXAMPLES=OFF"
    # Video reading and writing on Linux
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_V4L=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON"
    # Videoio plugins
    export CMAKE_ARGS="${CMAKE_ARGS} -D VIDEOIO_ENABLE_PLUGINS=ON"
    # Parallel processing
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_PTHREADS_PF=ON"
    # Thrrading plugins
    export CMAKE_ARGS="${CMAKE_ARGS} -D PARALLEL_ENABLE_PLUGINS=ON"
    # GUI Backends
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_GTK=ON -D WITH_QT=ON"
    # OpenGL
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_OPENGL=ON"
    # highgui plugins
    export CMAKE_ARGS="${CMAKE_ARGS} -D HIGHGUI_ENABLE_PLUGINS=ON"
    # Deep learning neural networks inference backends and options (dnn module)
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_PROTOBUF=ON -D BUILD_PROTOBUF=ON -D OPENCV_DNN_OPENCL=ON -D OPENCV_DNN_CUDA=ON"
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_VULKAN=OFF" # Disable experimental Vulkan backend for now
    
    # Change Installation root, if necessary
    #export CMAKE_ARGS="${CMAKE_ARGS} -D CMAKE_INSTALL_PREFIX=<whatever-path>"
    
    # Install Python sample sources from the samples/python directory.
    export CMAKE_ARGS="${CMAKE_ARGS} -D INSTALL_PYTHON_EXAMPLES=OFF" 
    # Miscellaneous features
    export CMAKE_ARGS="${CMAKE_ARGS} -D OPENCV_ENABLE_NONFREE=ON -D ENABLE_CCACHE=ON -D BUILD_DOCS=OFF"
    export CMAKE_ARGS="${CMAKE_ARGS} -D ENABLE_PYLINT=ON -D ENABLE_FLAKE8=ON" # Python linters
    export CMAKE_ARGS="${CMAKE_ARGS} -D BUILD_opencv_python3=ON -D BUILD_opencv_python2=ON" # Python bindings
    # Contrib Modules
    export CMAKE_ARGS="${CMAKE_ARGS} -D WITH_CLP=OFF"

    # Enable building opencv-contrib-python package
    export ENABLE_CONTRIB=1

    # Upgrade build tools
    python3 pip install --upgrade pip setuptools wheel
    # Build!
    python3 -m pip wheel . --verbose
    
    echo -n "======================================================================"
    echo -n "|   The OPENCV-CUDA install script has completed its execution.      |"
    echo -n "======================================================================"
}

install_yolo () {
    cd "$PROJECT_PATH" || exit
    check_python_venv "$(pwd)"
    
    python3 -m pip -v install git+https://github.com/sunsmarterjie/yolov12.git

    echo -n "==============================================================="
    echo -n "|   The YOLO install script has completed its execution.      |"
    echo -n "==============================================================="
}

install_opencv_cuda
install_pypi_packages
install_yolo