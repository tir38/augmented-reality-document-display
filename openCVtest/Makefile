# makefile for openCVtest project
# Jason Atwood 6/18/2013

CC=g++ 						# set compiler to g++
CFLAGS=`pkg-config --cflags opencv`		# opencv compiler flags
LIBS=`pkg-config --libs opencv`			# opencv libraries 
	
# project name : list of .cc files
OCVtest: openCVtest.cc					
	$(CC) $(CFLAGS) -o OCVtest openCVtest.cc $(LIBS) 
