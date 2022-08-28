CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3-config --cflags)
##如果第一句报错，则将 $(shell python3-config --cflags)换成python的环境路径 -I/Users/gwzhome/opt/anaconda3/envs/tf2/include/python3.6m
## 该路径可以通过执行python3-config --cflags命令行获得
LDFLAGS = $(shell python3-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = pixelmerge.cpp include/clipper/clipper.cpp
OPENCV = `pkg-config --cflags --libs opencv`

LIB_SO = pixelmerge.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)


