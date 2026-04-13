MDC ?= cmark

MDDIR = md
HTMLDIR = html

MDFILES = $(wildcard $(MDDIR)/*.md)
HTMLFILES = $(MDFILES:$(MDDIR)/%.md=$(HTMLDIR)/%.html)

.PHONY: all
all: $(HTMLFILES)

$(HTMLFILES): $(HTMLDIR)/%.html: $(MDDIR)/%.md
	@mkdir -p $(HTMLDIR)
	cat start.html <(head -n1 $<) middle.html <($(MDC) --unsafe <(tail -n+2 $<)) end.html > $@

$(HTMLFILES): start.html middle.html end.html
