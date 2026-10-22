# Documentation build.
#   make diagrams   render diagrams/*.dot to SVG (commit the SVGs)
#   make html       diagrams + Doxygen API reference in _build/html
#   make clean

DOT_SOURCES := $(wildcard diagrams/*.dot)
SVGS := $(DOT_SOURCES:.dot=.svg)

.PHONY: all diagrams html clean
all: html

diagrams: $(SVGS)

diagrams/%.svg: diagrams/%.dot
	dot -Tsvg $< -o $@

html: diagrams
	cd .. && doxygen Doxyfile

clean:
	rm -rf _build
