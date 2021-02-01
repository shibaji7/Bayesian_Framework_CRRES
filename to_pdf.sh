jupyter nbconvert --to latex $1.ipynb
pdflatex $1
rm $1.tex; rm $1.out; rm $1.aux; rm $1.log
rm -rf $1_files
