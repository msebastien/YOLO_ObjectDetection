#!/usr/bin/env bash
# Script to install v4l2loopback for DroidCam which allows using an Android phone as a webcam
# Written by Sébastien Maes

VERSION=0.14.0

# Download and extract the tarball (tar requires superuser privileges)
curl -L https://github.com/umlaeute/v4l2loopback/archive/v${VERSION}.tar.gz | sudo tar xvz -C /usr/src

# Install kernel module using DKMS
sudo dkms add -m v4l2loopback -v ${VERSION}
sudo dkms build -m v4l2loopback -v ${VERSION}
sudo dkms install -m v4l2loopback -v ${VERSION}

# Install v4l2loopback-ctl utility program
cd /usr/src/v4l2loopback-${VERSION}/utils && make && sudo make install

# Load the module at boot
if [ ! -d /etc/modules-load.d/v4l2loopback.conf ]; then
	cd /etc/modules-load.d && sudo touch v4l2loopback.conf
	echo "v4l2loopback" > v4l2loopback.conf
	cd $HOME
fi

