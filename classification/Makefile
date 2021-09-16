.PHONY: clean

TIDL_API_DIR := /usr/share/ti/tidl/tidl_api
LD            = g++ -o
LDFLAGS       = $(TIDL_API_DIR)/tidl_api.a \
                -lopencv_imgproc \
                $(TIDL_API_DIR)/tidl_imgutil.a \
                -lTIOpenCL \
                -lpthread
CXX           = g++ -c
CXXFLAGS      = -fPIC -O3 -Wall -Werror -Wno-error=ignored-attributes \
                -I$(TIDL_API_DIR)/inc -std=c++14 -I/usr/share/ti/opencl 

classification.so:

%.so: %.o
	$(LD) $@ -shared $^ $(LDFLAGS) 

clean:
	rm -f classification.so
