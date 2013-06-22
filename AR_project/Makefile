# makefile for augmented reailty project
# Jason Atwood 6/22/2013

CC=g++ 						# set compiler to g++
CFLAGS=`pkg-config --cflags opencv`		# opencv compiler flags
LIBS=`pkg-config --libs opencv`			# opencv libraries 
	
# project name : list of .cc files
ARproject: mainAR.cc					
	$(CC) $(CFLAGS) -o ARproject mainAR.cc $(LIBS) 
