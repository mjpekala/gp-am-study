#


default :
	@echo "please pick a report to publish"

nist-m1 :
	\rm -rf ./nist-m1_files
	\rm -f nist-m1.pdf
	ipython nbconvert --to pdf --template latex_nocode NIST_M290_maraging_steel.ipynb


gp1d :
	\rm -rf ./gp1d_files
	\rm -f gp1d.pdf
	ipython nbconvert --to pdf --template latex_nocode gp1d.ipynb


rotate :
	ipython nbconvert --to pdf --template latex_nocode rotation_question.ipynb
	
