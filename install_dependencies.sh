#!/usr/bin/env bash
# Written by Sébastien Maes

IS_VENV_INITIALIZED=false
PROJECT_DIRECTORY_NAME=yolov12-handgestures

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
    cd "$HOME"/"$PROJECT_DIRECTORY_NAME" || exit
    check_python_venv "$(pwd)"

    python3 -m pip install -U -v    \
    opencv-python                   \
    opencv-contrib-python           \
    numpy                           \
    pysdl2                          \
    pysdl2-dll                      \
    huggingface-hub                 \
    supervision

    echo -n "========================================================================"
    echo -n "|   The PyPI packages install script has completed its execution.      |"
    echo -n "========================================================================"
}

install_yolo () {
    cd "$HOME"/"$PROJECT_DIRECTORY_NAME" || exit
    check_python_venv "$(pwd)"
    
    python3 -m pip -v install git+https://github.com/sunsmarterjie/yolov12.git

    echo -n "==============================================================="
    echo -n "|   The YOLO install script has completed its execution.      |"
    echo -n "==============================================================="
}

install_pypi_packages
install_yolo