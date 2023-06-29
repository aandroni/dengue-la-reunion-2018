CC = g++
CFLAGS = -O2 -Wall -std=gnu++14
LDFLAGS =
INCLUDES =

SOURCES = main.cpp mcmc.cpp varia.cpp model.cpp
HEADERS = mcmc.hpp varia.hpp model.hpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = dengue.exe

test: $(EXECUTABLE) $(SOURCES) $(HEADERS)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(INCLUDES) $(OBJECTS) -o $@

%.o: %.cpp $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean

clean:
	rm -rf *.o *.exe
