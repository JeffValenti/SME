# This is the general procedure for setting NLTE corrections wherever possible.
#
# Corrections are read from a generalized NLTE procedure, defined by sme.nlte_pro
# (currently: "sme_nlte.pro")
#
# The input SME structure should be modified to include the following fields:
#	nlte_pro: The procedure to call, which handles all departure coefficients.
#	nlte_elem_flags: bytarr(99) flags which elements should be treated under NLTE.
#	nlte_grids: strarr(99) paths to grids of departure coefficients
#	[nlte_subgrid_size]: intarr(4) (optional) number of grid points to keep for each parameter: [X/Fe], Teff, logg, [Fe/H]
# Additionally, it should contain fields relevant to a VALD3 long-formatted
#  linelist, to enable identification of atomic levels.
# Only spectral lines where corrections are available for both upper and lower
#  atomic levels will be corrected.

import numpy as np
import .sme_synth

def sme_update_depcoeffs(sme, clear = False, debug=False):
    # @NLTE/common_nlte
    # common entrypoints, entry, prefix, sme_library # for communication with library
    # common sme_main_func, atmo, vmac # required to determine number of depth points in atmosphere

    # # Common block keeps track of the currently stored subgrid, 
    # #  i.e. that which surrounds a previously used grid points, 
    # #  as well as the current matrix of departure coefficients.


    ## Reset departure coefficients from any previous call, to ensure LTE as default:
    #errstr = call_external(sme_library, entry.resetNLTE, /s_value)
    #if errstr ne '' then message, 'ResetDepartureCoefficients (call_external): ' + errstr

    atmo = sme.atmo
    ndepths = len(atmo.temp)
    if not "nlte" in sme:
        return sme # no NLTE is requested

    if not "nlte" in sme or not "nlte_pro" in sme.nlte or len(sme.nlte.nlte_pro) == 0 or not "nlte_elem_flags" in sme.nlte or np.sum(sme.nlte.nlte_elem_flags) == 0 or not 'nlte_grids' in sme.nlte or np.all(sme.nlte.nlte_grids == ''):
        # Silent fail to do LTE only.
        if debug:
            print('Running in LTE')
        return # no NLTE routine available
    elif no 'line_extra' in sme:
        print('--- ')
        print('NLTE line formation was requested, but VALD3 long-format linedata ')
        print('are required in order to relate line terms to NLTE level corrections!')
        print('Line formation will proceed under LTE.')
        return # no NLTE line data available
    if debug:
        print('Running in NLTE')

    if clear:
        #???
        if sme.nlte.nlte_grid > 0:
            ptr_free, nlte_grid, nlte_linelevelrefs, nlte_lineindices
            if nlte_debug then print, 'Clearing previous NLTE grid'
        clear = True # ok, already cleared



    if len(nlte_b) < 2: # first call during run
        if debug:
            print('Initializing main NLTE storage')
        firstcall = 1 # initialize storage vectors
        
        # All NLTE coefficients are stored in this matrix:
        nlte_b = np.ones((ndepths, len(sme.species), 2))
        nlte_grid = np.zeros(99)
        # Correlate between nlte_b and corrections for each species, nlte_lev_b
        nlte_linerefs = np.zeros((len(sme.species), 2), dtype=int)
        nlte_linelevelrefs = np.zeros(99)
        nlte_lineindices   = np.zeros(99)

    # Call each element to update and return its set of departure coefficients
    i = 0
    updated = 0
    tmp = 0
    for elem in range(len(sme.nlte.nlte_elem_flags)):
        if sme.nlte.nlte_elem_flags[elem]:
            # Call function to retrieve interpolated NLTE departure coefficients
            bmat = sme.nlte.nlte_pro(sme, elem, blevels, linerefs, lineindices, updated=updated, present=present)
            
            if len(linerefs) < 2 or updated is None or present is None:
                # no data were returned. Don't bother?
                pass
            else:
                # Put corrections into the nlte_b matrix, don't cache the data
                for l in range(len(lineindices)):
                    # loop through the list of relevant _lines_, substitute both their levels into the main b matrix
                    # Make sure both levels have corrections available
                    if linerefs[l, 0] != -1 and linerefs[l, 1] != -1:
                        errstr = sme_synth.InputNLTE(bmat[linerefs[l,:], :].T, lineindices[l])

    if not updated:
        return # the previous b-matrix is still valid

    # DEBUG:
    if debug:
        print(' ---------------- Setting departure coefficients ----------------')
    ## Visualize corrections for all levels:
    if debug:
        pass
        #     ps_open, 'nlte_debug', /color, /ps
        #     uspec = sme.nlte.species & uspec = uspec[uniq(uspec, sort(uspec))]
        #     colors = c24([2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        #     for us=0,len(uspec)-1 do begin
        #         ii = where(sme.nlte.species eq uspec[us] and nlte_b[0,*,1] ne 1, nii)
        #         if nii eq 0 then continue
        #         for l=0,nii-1 do begin
        #             if 0 eq (l mod 40) then begin
        #                 if l gt 0 then legend, leg[1:*], colors=colors[indgen(nii) mod len(colors)], linestyle=0, /right
        #                 plot, [0,55], [1,1], yr=minmax(nlte_b[*,ii,*]), ys=3, title=uspec[us], linestyle=1
        #                 leg = '' & it = 0L
        #             endif
        #             leg = [leg, strtrim(sme.nlte.atomic[2,ii[l]],2)]
        #             for i=0,1 do begin
        #                 oplot, nlte_b[i,ii[l],*], col=colors[l mod len(colors)], linestyle=i
        #             endfor
        #         endfor
        #         legend, leg[1:*], colors=colors[indgen(nii) mod len(colors)], linestyle=0, /right
        #     endfor
        #     ps_close
        #     stop
        # endif

    return sme





