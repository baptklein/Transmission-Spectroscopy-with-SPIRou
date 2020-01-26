# An introduction to transmission spectroscopy with SPIRou
## EXOSYSTÈMES I - ÉVOLUTION - 30/01/2019

This tutorial is aimed at acquainting the user with SPIRou data analysis in the context of transmission spectroscopy.
Based on a publicly available sequence of spectra of Gl514 obtained with SPIRou on 2019 May 14 (ID: 19AE99, PI: Claire Moutou),
the associated jupyter notebook deals one by one with all the steps required to clean the spectra and carry out the seach of
water absorption signatures from the planet atmosphere during a synthetic transit event. 


## Getting Started

To run this tutorial on your local machine, please copy this github directory into your local machine and execute the file
"Main.ipynb" using **jupyter notebook**. All the instructions required for this tutorial are stored therein. If you do not have
jupyter notebook, a python version is also provided "main.py", but it will be less convenient to use.


### Prerequisites

The following programs are needed to make the code work:
1 - python3 (code implemented on python 3.5.2)
2 - jupyter notebook 
    Installed via pip: pip install jupyterlab
    or conda. See https://jupyter.org/install for proper installation procedure
    To run the notebook open a terminal in the github directory containing this tutorial (downloaded)
    Type 'jupyter notebook' command to run the module. It opens a new window in your browser.
    Open "Main.ipynb" and start the tutorial.
    
The following python modules are required (in addition to standard python modules)
  - scipy
  - termcolor (pip install termcolor)
  - astropy
  - batman (pip install batman-package)
  - sklearn (pip install -U scikit-learn)
  


  
  
## Authors

* **Baptiste Klein**
* **Claire Moutou**
* **Jean-François Donati**
from IRAP, Toulouse

  
  
## Acknowledgments
We warmly thank the organising committee for giving us the opportunity to share this tutorial and for this successful conference.
We also thank the local organising committee for the flawless organisation.
Last but not least, we thank all the users and will be happy to have their feedbacks.
