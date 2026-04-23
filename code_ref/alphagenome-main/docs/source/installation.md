# Installation

The easiest way to install AlphaGenome is via the published
[PyPi package](https://pypi.org/project/alphagenome).

```bash
$ pip install -U alphagenome
```

This will install the latest version of the `alphagenome` package.

You may optionally wish to create a
[Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) to
prevent conflicts with your system's Python environment.

## Google Colab

The tutorial notebooks include a cell with the commands necessary to install
`alphagenome` into a Colab runtime.

### Add API key to secrets

To make model requests using the tutorial notebooks, you need to add the
AlphaGenome API key to Colab secrets:

1.  Open your Google Colab notebook and click on the 🔑 **Secrets** tab in the
    left panel.
1.  Create a new secret with the name `ALPHA_GENOME_API_KEY`.
1.  Copy/paste your API key into the `Value` input box of
    `ALPHA_GENOME_API_KEY`.
1.  Toggle the button on the left to allow notebook access to the secret.

<!-- mdformat off(Turn off mdformat to retain myst syntax.) -->

```{figure} /_static/secrets.png
:width: 600px
:alt: Image of secrets tab found on left panel.
:name: secrets-screenshot
```
<!-- mdformat on -->

## Running locally

To install a local copy of `alphagenome`, clone a local copy of the repository
and run `pip install`:

```bash
$ rm -rf ./alphagenome
$ git clone https://github.com/google-deepmind/alphagenome.git
$ pip install -e ./alphagenome
```

We strongly recommend using a virtual environment management system such as
[miniconda](https://docs.anaconda.com/miniconda/) or
[uv](https://docs.astral.sh/uv/pip/environments/).

In the case of miniconda, installation would be achieved with the following:

```bash
conda create -n alphagenome-env python=3.11
conda activate alphagenome-env
pip install -e ./alphagenome
```

### Updating `alphagenome`

Assuming the relevant virtual environment is already activated:

```bash
cd ./alphagenome
git pull
pip install --upgrade .
```
