#


default :
	@echo "please pick a report to publish"

all : apl-maraging pins-maraging


apl-maraging :
	ipython nbconvert --to pdf --template latex_nocode APL_M290_maraging_steel.ipynb

pins-maraging :
	ipython nbconvert --to pdf --template latex_nocode PINS01_M290_maraging_steel.ipynb

clean:
	\rm -rf ./APL_M290_maraging_steel_files
	\rm -f ./APL_M290_maraging_steel.pdf
	\rm -rf ./PINS01_M290_maraging_steel_files
	\rm -f ./PINS01_M290_maraging_steel.pdf
