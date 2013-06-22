# makefile for openCVtest project
# Jason Atwood 6/18/2013

CC=g++ 				# set compiler to g++
CFLAGS=-I.			#  
	
openCVtest: openCVtest.cc					# project name : list of .cc files
	$(CC)  -o openCVtestExecutable openCVtest.cc $(CFLAGS)	# $(compiler) $(compiler flags) executable name, -o files $(libraries)
