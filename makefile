.PHONY: install

install-cpu:
	pip install -e .
	pip install torch-scatter -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cpu.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cpu.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cpu.html


install-cu118:
	pip install -e .
	pip install torch-scatter -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cu118.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cu118.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cu118.html


install-cu121:
	pip install -e .
	pip install torch-scatter -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cu121.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cu121.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-$(shell pip show torch | grep Version | sed "s/Version: //g")+cu121.html

