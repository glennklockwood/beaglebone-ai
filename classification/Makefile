.PHONY: clean

TIDL_API_DIR := /usr/share/ti/tidl/tidl_api
LD            = g++ -o
LDFLAGS       = -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio \
                -lopencv_imgproc -lopencv_core $(TIDL_API_DIR)/tidl_api.a \
                $(TIDL_API_DIR)/tidl_imgutil.a -lTIOpenCL -locl_util -lpthread
CXX           = g++ -c -o
CXXFLAGS      = -fPIC -O3 -Wall -Werror -Wno-error=ignored-attributes -I. \
                -I$(TIDL_API_DIR)/inc -std=c++14 -I/usr/share/ti/opencl 

classification.so:

%.so: %.o
	$(LD) $@ -shared $^ $(LDFLAGS) 

clean:
	rm -f classification.so
