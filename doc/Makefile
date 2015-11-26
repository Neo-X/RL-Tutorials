# make -f Makefile

texDir = tex
R = project

pdf:
	make -C $(texDir) pdf

clean:
	make -C $(texDir)  clean
	rm -f $(R).log $(R).aux $(R).bbl $(R).blg $(R).out $(R).pdf $(R).synctex.gz
