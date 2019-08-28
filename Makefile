define COMPILEXX
	@echo "CXX: $1"
	$(Q) $(CXX) -c $(CXXFLAGS) $1 -o $2 $(DEBUGFLAGS)
endef

define ARCHIVE
	@echo "ARCHIVE: $1"
	$(Q) $(CXX) -o $1 $2 $(OPTIMIZATION) $(LDFLAGS) $(DEBUGFLAGS)
endef

define CLEAN
	@echo "CLEAN"
	$(Q) rm -f *$(OBJEXT) $(BIN) *~ .*.swp .built
endef

export Q := @
CXX = g++

OPTIMIZATION = -O3
CXXFLAGS = -fPIC -std=c++11 $(OPTIMIZATION)

OBJEXT = .o

CXXSRCS =
CXXOBJS = $(CXXSRCS:.cpp=$(OBJEXT))

OBJS = $(CXXOBJS)

BIN = mnist

SOPHON_PATH := /home/jseong/sophon/release
INCLUDE_DIRS = $(SOPHON_PATH)/include
INCLUDE_DIRS += $(SOPHON_PATH)/include/bmlib
INCLUDE_DIRS += $(SOPHON_PATH)/include/bmdnn
INCLUDE_DIRS += $(SOPHON_PATH)/include/bmruntime

CXXFLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))

LIB_DIRS = $(SOPHON_PATH)/lib/sys
LIB_DIRS += $(SOPHON_PATH)/lib/app
LDFLAGS += $(foreach libdir,$(LIB_DIRS),-L$(libdir))

LIBS = stdc++
LIBS = bmdnn_device bmlib_device
LDFLAGS += $(foreach libs,$(LIBS),-l$(libs))

DEBUGFLAGS = -g -Ddebug

VPATH = .

include src/Make.defs

all: .built
.PHONY: .built networks clean

.built: $(BIN)
	$(Q) touch .built

$(BIN): $(OBJS)
	$(call ARCHIVE, $(BIN), $(OBJS))

$(CXXOBJS): %$(OBJEXT): %.cpp
	$(call COMPILEXX, $<, $@)

clean:
	$(call CLEAN)
