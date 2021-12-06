# torch_constrained

## Steps for development

* Python >= 3.8

### 1) Clone the repository with the ssh link

```git clone git@github.com:manuel-delverme/constrained_optimization.git ```

### 2) `venv` and requirement installation

```angular2html
install torch manually for your cuda driver https://pytorch.org/get-started/locally/

python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt

```

### 3) Enable flake for good code checking

```
pre-commit install
git config --bool flake8.strict true
```
