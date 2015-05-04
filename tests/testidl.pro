Pro testidl
  readcol, 'test.dat', x, y, xsig, ysig
  linmix_err, x, y, post, xsig=xsig, ysig=ysig
  openw, lun, 'test.idlout', /get_lun
  printf, lun, 'alpha beta sigsqr pi0 pi1 pi2 mu0 mu1 mu2 tausqr0 tausqr1 tausqr2 mu00 usqr wsqr ximean xisig corr'
  writecol, 'test.out', post.alpha, post.beta, post.sigsqr, $
            post.pi[0], post.pi[1], post.pi[2], $
            post.mu[0], post.mu[1], post.mu[2], $
            post.tausqr[0], post.tausqr[1], post.tausqr[2], $
            post.mu0, post.usqr, post.wsqr, $
            post.ximean, post.xisig, post.corr, $
            fmt='(f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f)', $
            filnum = lun
  close, lun
end
