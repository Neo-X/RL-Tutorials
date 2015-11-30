# make -f Makefile_pdflatex

R = project
STY = *.cls
FIG = ../images/*.png

pdf: $(R).pdf

bib: $(R).bib
	pdflatex $(R)
	bibtex $(R)
	pdflatex $(R)

$(R).pdf: *.tex $(FIG) bib
	pdflatex $(R)
	cp $(R).pdf ../.

clean:
	rm -f $(R).log $(R).aux $(R).bbl $(R).blg $(R).out $(R).pdf $(R).synctex.gz $(R).toc