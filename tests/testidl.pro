Pro testidl
  readcol, 'test.dat', x, y, xsig, ysig, delta
  linmix_err, x, y, post, xsig=xsig, ysig=ysig, delta=delta
  openw, lun, 'test.idlout', /get_lun
  printf, lun, 'alpha beta sigsqr mu00 usqr wsqr ximean xisig corr'
  writecol, 'junk', post.alpha, post.beta, post.sigsqr, $
            post.mu0, post.usqr, post.wsqr, $
            post.ximean, post.xisig, post.corr, $
            fmt='(9f12.5)', $
            filnum = lun
  close, lun
end
