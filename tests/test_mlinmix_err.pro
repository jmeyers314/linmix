Pro test_mlinmix_err
  readcol, 'test_mlinmix.dat', x1, x2, y, x1var, x2var, yvar
  x = [[x1], [x2]]
  nx = n_elements(x1)
  np = 2
  xvar = dblarr(nx, np, np)
  for i=0, nx-1 do begin
      xvar[i,*,*] = diag_matrix([x1var[i], x2var[i]])
  endfor
  mlinmix_err, x, y, post, xvar=xvar, yvar=yvar
  openw, lun, 'test_mlinmix.idlout', /get_lun
  printf, lun, 'alpha beta1 beta2'
  writecol, 'junk', post.alpha, post.beta[0], post.beta[1], post.sigsqr, $
            fmt='(4f12.5)', $
            filnum = lun
  close, lun
  stop
end
