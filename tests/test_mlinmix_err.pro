Pro testidl
  readcol, 'test_mlinmix.dat', x1, x2, y, x1var, x2var, yvar
  x = [x1, x2]
  xvar = [x1var, x2var]
  mlinmix_err, x, y, post, xvar=xvar, yvar=yvar
  ; openw, lun, 'test.idlout', /get_lun
  ; printf, lun, 'alpha beta sigsqr mu00 usqr wsqr ximean xisig corr'
  ; writecol, 'junk', post.alpha, post.beta, post.sigsqr, $
  ;           post.mu0, post.usqr, post.wsqr, $
  ;           post.ximean, post.xisig, post.corr, $
  ;           fmt='(9f12.5)', $
  ;           filnum = lun
  ; close, lun
  stop
end
