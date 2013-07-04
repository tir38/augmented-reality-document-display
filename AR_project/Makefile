# makefile for augmented reailty project
# Jason Atwood 6/22/2013

CC=g++												# set compiler to g++
CFLAGS=`pkg-config --cflags opencv, Magick++`		# compiler flags
LIBS=`pkg-config --libs opencv, zbar, Magick++`		# libraries
	
# project name : list of .cc files
ARproject: mainAR.cc helpers.cc
	$(CC) $(CFLAGS) -o ARproject mainAR.cc helpers.cc $(LIBS)
