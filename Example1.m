% script reproducing some of the results displayed in Example 1 of the paper:
% S.Gazzola and M.Sabate Landman. 
% Krylov Methods for Inverse Problems: surveying classical, and introducing new, algorithmic approaches.
% To appear in GAMM-Mitteilungen

% Silvia Gazzola, University of Bath
% Malena Sabate Landman, University of Bath
% June, 2020.

%% generating the test problem
path(path, './ExtraBilevel')

load('X_544.mat')
% subimage of a galaxy image available at 
% https://public.nrao.edu/gallery/topic/galaxies/

%%% Blurring forward problem
PSF = psfNSGauss([32, 32], 6, 5, 2.5);
PSF = PSF/sum(PSF(:));
optblur.PSF = PSF;
optblur.trueImage=X;
% avoid crime
[A, b, x, ProbInfo] = PRblur(optblur);
n = ProbInfo.xSize(1);

%%% Add some noise to the data
rng(30)
originalnl = 5e-2;
bn = PRnoise(b, originalnl);

%%% display true image and data
figure, imagesc(reshape(x,n,n)), axis image, axis off
title('Sharp Image')
figure, imagesc(reshape(b,n,n)), axis image, axis off
title('Blurred Image')

%% set some parameters for the solvers
optsolver = IRhybrid_lsqr('defaults');
optsolver.x_true = x;
optsolver.NoStop = 'on';
optsolver.eta = 1;
optsolver.NoiseLevel = originalnl; 
maxiter = 100;
K = [1, 10:10:maxiter]; 

% run LSQR
optsolver.RegParam = 0; 
[X_lsqr, info_lsqr] = IRhybrid_lsqr(A, bn, K, optsolver); 

% run hybrid LSQR, with the discrepancy principle
optsolver.RegParam = 'discrepit';
optsolver.plotty = 'on';
[X_hlsqr_dp, info_hlsqr_dp] = IRhybrid_lsqr(A, bn, K, optsolver); 

% run the new bilevel adaptive parameter choice method
% with the discrepancy principle
optsolver.RegParam0=1e10;
optsolver.RegParam = 'discrepbil';
optsolver.discrbilStopTol = 1e-3;
optsolver.RegParamRange = [1e-6, 1e2];
[X_hlsqr_dpbil, info_hlsqr_dpbil] = IRhybrid_lsqr(A, bn, K, optsolver); 

% run hybrid LSQR, with the optimal regularization parameter at each iteration
optsolver.RegParam = 'optimal';
[X_hlsqr_opt, info_hlsqr_opt] = IRhybrid_lsqr(A, bn, K, optsolver); 

% run hybrid LSQR, with the Reginska criterion
optsolver.RegParam = 'reginskait'; % 
[X_hlsqr_regn, info_hlsqr_regn] = IRhybrid_lsqr(A, bn, K, optsolver); 

% run the new bilevel adaptive parameter choice method
% with Reginska criterion
optsolver.RegParam0= 1e-3;
optsolver.regbilStopTol = 1e-1;
optsolver.RegParam = 'reginskabil';
optsolver.RegParRegRange = [1e-10, 1e0];
[X_hlsqr_regnbil, info_hlsqr_regnbil] = IRhybrid_lsqr(A, bn, K, optsolver); 

%% produce a few plots
% upper and lower bounds for the discrapancy curves
figure
loglog(info_hlsqr_dpbil.RegPrange, info_hlsqr_dpbil.DPcurvelow(1,:)' + (1.05*originalnl*norm(bn))^2, '-b')
hold on
loglog(info_hlsqr_dpbil.RegPrange, info_hlsqr_dpbil.DPcurvelow(3,:)' + (1.05*originalnl*norm(bn))^2, '-r')
loglog(info_hlsqr_dpbil.RegPrange, info_hlsqr_dpbil.DPcurvelow(15,:)' + (1.05*originalnl*norm(bn))^2, '-m')
loglog(info_hlsqr_dpbil.RegPrange, info_hlsqr_dpbil.DPcurveup(1,:)' + (1.05*originalnl*norm(bn))^2, '-b')
loglog(info_hlsqr_dpbil.RegPrange, info_hlsqr_dpbil.DPcurveup(3,:)' + (1.05*originalnl*norm(bn))^2, '-r')
loglog(info_hlsqr_dpbil.RegPrange, info_hlsqr_dpbil.DPcurveup(15,:)' + (1.05*originalnl*norm(bn))^2, '-m')
legend('k=1', 'k=3', 'k=15')
xlabel('\alpha')
ylabel('discrepancy function')
% upper and lower bounds for the Reginska curves
figure
loglog(info_hlsqr_regnbil.RegPrange, info_hlsqr_regnbil.ReginUp(3,:)', '-b')
hold on
loglog(info_hlsqr_regnbil.RegPrange, info_hlsqr_regnbil.ReginUp(7,:)', '-r')
loglog(info_hlsqr_regnbil.RegPrange, info_hlsqr_regnbil.ReginUp(15,:)', '-m')
loglog(info_hlsqr_regnbil.RegPrange, info_hlsqr_regnbil.ReginLow(3,:)', '-b')
loglog(info_hlsqr_regnbil.RegPrange, info_hlsqr_regnbil.ReginLow(7,:)', '-r')
loglog(info_hlsqr_regnbil.RegPrange, info_hlsqr_regnbil.ReginLow(15,:)', '-m')
legend('k=3', 'k=7', 'k=15')
xlabel('\alpha')
ylabel('Reginska function')
% regularization parameter history
figure
semilogy(info_hlsqr_dpbil.RegP, '-b')
hold on
semilogy(info_hlsqr_dp.RegP, '-.b')
semilogy(info_hlsqr_regnbil.RegP, '-r')
semilogy(info_hlsqr_regn.RegP, '-.r')
semilogy(info_hlsqr_opt.RegP, '-k')
legend('DP', 'new DP', 'Reginska', 'new Reginska', 'optimal')
xlabel('iteration counter')
ylabel('regularization parameter')
% relative error history
figure
semilogy(info_hlsqr_dpbil.Enrm, '-b')
hold on
semilogy(info_hlsqr_dp.Enrm, '-.b')
semilogy(info_hlsqr_regn.Enrm, '-r')
semilogy(info_hlsqr_regnbil.Enrm, '-.r')
semilogy(info_hlsqr_opt.Enrm, '-k')
legend('DP', 'new DP', 'Reginska', 'new Reginska', 'optimal')
xlabel('iteration counter')
ylabel('relative error')