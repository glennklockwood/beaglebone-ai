.PHONY: clean

TIDL_API_DIR := /usr/share/ti/tidl/tidl_api
CXX           = g++
CXXFLAGS      = -fPIC -O3 -std=c++14 \
                -Wall -Werror -Wno-error=ignored-attributes \
                -I$(TIDL_API_DIR)/inc -I/usr/share/ti/opencl
LDFLAGS       = -lopencv_imgproc -lTIOpenCL -lpthread

classification.so: classification.o $(TIDL_API_DIR)/tidl_api.a $(TIDL_API_DIR)/tidl_imgutil.a
	$(CXX) -o $@ -shared $^ $(LDFLAGS)

clean:
	rm -f classification.so *.o
