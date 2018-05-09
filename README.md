# SME
Spectroscopy Made Easy (SME) is a software tool that fits an observed
spectrum of a star with a model spectrum. Since its initial release in
[1996](http://adsabs.harvard.edu/abs/1996A%26AS..118..595V), SME has been a
suite of IDL routines that call a dynamically linked library, which is
compiled from C++ and fortran. This classic IDL version of SME is available
for [download](http://www.stsci.edu/~valenti/sme.html).

In 2018, we began began reimplmenting the IDL part of SME in python 3,
adopting an object oriented paradigm and continuous itegration practices
(code repository, build automation, self-testing, frequent builds).
** This project is very far from completion, so unless you want to help
the python development effort, use the IDL version. **
