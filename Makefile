CXX=hipcc
CXXFLAGS=-I. -I/opt/rocm/include -Wall
DEPS=helper_timer.h
OBJ=p2pBandwidthLatencyTest.o

%.o: %.cpp $(DEPS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

p2pBandwidthLatencyTest: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS)

.PHONY: clean run

clean:
	rm *.o p2pBandwidthLatencyTest

run: p2pBandwidthLatencyTest
	./p2pBandwidthLatencyTest
