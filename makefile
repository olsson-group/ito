.PHONY: install

install-cpu:
	pip install -e .
	pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cpu.html


install-cu118:
	pip install -e .
	pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html


install-cu121:
	pip install -e .
	pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
	pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
	pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

