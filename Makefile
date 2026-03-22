# Makefile


# INC      = $(PATH)/include/
CLANG      = /opt/homebrew/opt/llvm/bin/clang++
GCC 	   = g++
STD        = -std=c++20
OPT	   	   = -O3 -march=native -flto
# PRF        = -g #-fprofile-arcs -ftest-coverage 
BREW       = -I/usr/local/boost/include/ -I/opt/homebrew/Cellar/boost/1.87.0_1/include/ -L/opt/homebrew/opt/llvm/lib -I/opt/homebrew/opt/eigen/include/eigen3/ -I./optim/header_only_version -I./optim
# The following flags are only for debuging purposes
# WRN = -Wall -Wextra -Werror
# SAN = -fsanitize=address -fsanitize=undefined -fno-sanitize-recover	
# STL = -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -D_FORTIFY_SOURCE=2

% : %.cpp
	$(CLANG) $(STD) $(OPT) $(PRF) $(BREW)  -o $@ $< -lm -Xpreprocessor -fopenmp -lomp $(SAN) $(STL) $(WRN)
#  	$(GCC) $(STD) $(OPT) $(PRF) $(BREW)  -o $@ $< -lm -fopenmp $(SAN) $(STL) $(WRN)
