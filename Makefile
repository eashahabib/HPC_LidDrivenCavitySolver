default: mySolver


.PHONY: clean # Specify that 'clean' is not a real file
	target

clean:
clean:
	-rm -f *.o mySolver # Clean up (and ignore any errors)


CXX = mpicxx

OBJS = LidDrivenCavitySolver.o
LDLIBS = -lscalapack-openmpi -llapack -lblas

%.o : %.cpp
	$(CXX) -o $@ $<

mySolver: $(OBJS)
	$(CXX) -o $@ $^ $(LDLIBS)

all: mySolver