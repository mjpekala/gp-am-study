#


default :
	@echo "please pick a report to publish"

gp1d :
	\rm -rf ./gp1d_files
	\rm -f gp1d.pdf
	ipython nbconvert --to pdf --template latex_nocode gp1d.ipynb


rotate :
	ipython nbconvert --to pdf --template latex_nocode rotation_question.ipynb
	
