#


default :
	@echo "please pick a report to publish"

nist-maraging :
	ipython nbconvert --to pdf --template latex_nocode NIST_M290_maraging_steel.ipynb

clean:
	\rm -rf ./NIST_M290_maraging_steel_files
	\rm -f ./NIST_M290_maraging_steel.pdf
