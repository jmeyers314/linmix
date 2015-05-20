Installation
============

Obtain through GIT
------------------

The code is hosted on `github <https://github.com/jmeyers314/linmix>`_ so the easiest method to
get a copy of the code is with `git <http://git-scm.com>`_::

	git clone https://github.com/jmeyers314/linmix.git

This will create a new subdirectory `linmix` containing the latest stable version of the complete
package.

Experts who already have a `correctly configured github account
<https://help.github.com/articles/which-remote-url-should-i-use/#cloning-with-ssh>`_ might prefer
this alternative::

	git clone git@github.com:jmeyers314/linmix.git

Update with GIT
---------------

You can update your local copy of the package at any time using::

	cd linmix
	git update

Required Packages
-----------------

The following python package is required by this package:

* numpy

Installing
----------

You can use the normal setup.py routines to install the linmix package into your python site-packages
directory.::

  python setup.py install

or if you want to install into your $HOME directory::

  python setup.py install --user
