
function phi = SV_slope(h, mu, sig2, prior, phi)

phi0 = prior(1);
invVphi = prior(2);

%% Sample phi 
Xphi = h(1:end-1)-mu;
zphi = h(2:end)-mu;
Dphi = 1/(invVphi +  sum((Xphi.^2)./sig2(2:end)));
phihat = Dphi*(invVphi*phi0 + sum((Xphi.*zphi)./sig2(2:end)));
phic = phihat + sqrt(Dphi)*randn;
% Dphi_half = sqrt(Dphi);
% phic = phihat + Dphi_half*trandn((-1-phihat)/Dphi_half,(1-phihat)/Dphi_half);
if abs(phic)<(1-1e-100)
    phic_logprior = -0.5*((phic-phi0)^2)*invVphi;
    eta = zphi - phic * Xphi;
    eta2 = eta.^2;
    phic_loglike = -0.5*sum(eta2./sig2(2:end));
    phic_logprop = -0.5*((phic-phihat)^2)/Dphi; 

    phi_logprior = -0.5*((phi-phi0)^2)*invVphi;
    eta = zphi - phi * Xphi;
    eta2 = eta.^2;
    phi_loglike = -0.5*sum(eta2./sig2(2:end));
    phi_logprop = -0.5*((phi-phihat)^2)/Dphi;     
    
    logprob = (phic_logprior + phic_loglike - phic_logprop) - ...
        (phi_logprior + phi_loglike - phi_logprop);
    if log(rand) <= logprob 
        phi = phic;
    end
end

