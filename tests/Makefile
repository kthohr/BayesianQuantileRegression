# BQReg test build

ifeq ($(CXX),)
	CXX = g++
endif

ifeq ($(BQREG_CXX_STD),)
	CXX_STD = -std=c++14
else
	CXX_STD=$(BQREG_CXX_STD)
endif

ifeq ($(BQREG_TEST_BUILD),)
	OPT_FLAGS = -O3 -mcpu=native -fopenmp
else
	OPT_FLAGS = -g -O0 -Wall -Wextra
endif

# source directories
SDIR = .
HEADERS = -I$(SDIR)/../cpp/include -I$(EIGEN_INCLUDE_PATH) -I$(SDIR)/../extr/gcem/include -I$(SDIR)/../extr/stats/include

SOURCES_TESTS= $(wildcard $(SDIR)/*.cpp)
OBJECTS_TESTS= $(SOURCES_TESTS:.cpp=.test)

all: $(OBJECTS_TESTS)

$(SDIR)/%.test: $(SDIR)/%.cpp
	$(CXX) $(CXX_STD) $(OPT_FLAGS) $(HEADERS) $< -o $@ $(LIBS)

# cleanup
.PHONY: clean
clean:
	@rm -rf *.test *.gcov *.gcno *.gcda *.dSYM

# individual make calls

BQREG_MAKE_CALL= $(CXX) $(CXX_STD) $(OPT_FLAGS) $(HEADERS) $@.cpp -o $@.test $(LIBS)

median_reg:
	$(BQREG_MAKE_CALL)

# rand:
# 	$(BQREG_MAKE_CALL)