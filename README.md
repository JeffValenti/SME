# SME
Spectroscopy Made Easy (SME) is a software tool that fits an observed
spectrum of a star with a model spectrum. Since its initial release in
[1996](http://adsabs.harvard.edu/abs/1996A%26AS..118..595V), SME has been a
suite of IDL routines that call a dynamically linked library, which is
compiled from C++ and fortran. This classic IDL version of SME is available
for [download](http://www.stsci.edu/~valenti/sme.html).

In 2018, we began reimplmenting the IDL part of SME in python 3,
adopting an object oriented paradigm and continuous itegration practices
(code repository, build automation, self-testing, frequent builds).

In 2018, we began reimplmenting the IDL part of SME in python with
the goal of making the code on GitHub maintainable by the community.
The new python implementation will include unit tests that cover all code
and integration tests that cover target platforms and software stacks.
The new code will use object-oriented design, defining classes that
encapsulate parameters that must be self-consistent and providing
methods that control interactions with those parameters.

**This project is very far from completion, so use the IDL version
unless you want to contribute to the python development effort.**

Developer and user documentation available at
[readthedocs](https://sme.readthedocs.io/en/latest/index.html).
Continuous integration build results available at
[Travis CI](https://travis-ci.org/JeffValenti/SME/builds).

