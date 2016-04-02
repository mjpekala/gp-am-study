#


default :
	@echo "please pick a report to publish"

all : nist-maraging apl01-maraging


nist-maraging :
	ipython nbconvert --to pdf --template latex_nocode NIST_M290_maraging_steel.ipynb

apl01-maraging :
	ipython nbconvert --to pdf --template latex_nocode APL01_M290_maraging_steel.ipynb

clean:
	\rm -rf ./NIST_M290_maraging_steel_files
	\rm -f ./NIST_M290_maraging_steel.pdf
	\rm -rf ./APL01_M290_maraging_steel_files
	\rm -f ./APL01_M290_maraging_steel.pdf
