K = 1:100;
% K = [K, 121, 130:10:250, 300];
K = [K, 121, 130:10:250];
% K = 1:20;
opt.RegParam = 0;
opt.NoStop = 'on';

n = 256;
ProbInfo.problemType = 'tomography';
ProbInfo.xType = 'image2D';
ProbInfo.xType = 'image2D';
ProbInfo.xSize = [n, n]; 

load DataFull_256x180.mat

ProbInfo.bSize = size(m);
b = m(:);

opt.RegMatrix = 'Gradient2D';
[X_sn, info_sn] = IRhybrid_lsqr(A, b, K, opt);
opt.RegMatrix = 'Identity';
[X, info] = IRhybrid_lsqr(A, b, K, opt);

optdp = opt;
optdp.RegParam = 'discrepit';
optdp.NoiseLevel = info.Rnrm(100);
optdp.eta = 1.05;

[X_dp, info_dp] = IRhybrid_lsqr(A, b, K, optdp);
optdp.RegMatrix = 'Gradient2D';
[X_sn_dp, info_sn_dp] = IRhybrid_lsqr(A, b, [K, 150], optdp);

optdpbil = optdp;
optdpbil.RegParam = 'discrepbil';
optdpbil.RegMatrix = 'Identity';
optdpbil.RegParam0 = 1e10;
[X_dpbil, info_dpbil] = IRhybrid_lsqr(A, b, K, optdpbil);
% 
optdpbil.RegMatrix = 'Gradient2D';
[X_sn_dpbil, info_sn_dpbil] = IRhybrid_lsqr(A, b, K, optdpbil);

optreg = opt;
optreg.RegParam = 'reginskait';
optreg.reginskaExp = 1; % 1.8; %% better with 0.7
[X_reg, info_reg] = IRhybrid_lsqr(A, b, K, optreg);
optreg.RegMatrix = 'Gradient2D';
optreg.reginskaExp = 1; 
[X_sn_reg, info_sn_reg] = IRhybrid_lsqr(A, b, K, optreg);

optregbil = optreg;
optregbil.RegParam = 'reginskabil';
optregbil.RegMatrix = 'Identity';
optregbil.RegParam0 = 1e1; % optregbil.RegParam0 = 1e0; % optregbil.RegParam0 = 1e-2; % 
[X_regbil, info_regbil] = IRhybrid_lsqr(A, b, K, optregbil);
optregbil.RegMatrix = 'Gradient2D';
optregbil.RegParam0 = 1e1; % optregbil.RegParam0 = 1e0; % optregbil.RegParam0 = 1e-2; % 
[X_sn_regbil, info_sn_regbil] = IRhybrid_lsqr(A, b, K, optregbil);

% %% Plotting the regularization parameters
% figure, semilogy(info_reg.RegP), hold on, semilogy(info_regbil.RegP)
% figure, semilogy(info_sn_reg.RegP), hold on, semilogy(info_sn_regbil.RegP)
% 
% %% Plotting bounds
% figure
% for i = [1:30, 31:5:80, 81:100]
% plot(info_dpbil.DPcurvelow(i,:), '-b'), hold on, plot(info_dpbil.DPcurveup(i,:), '-r'), plot(info_dp.DPcurveup(i,:), '-g')
% plot(info_dpbil.dots(i,1), info_dpbil.dots(i,2), 'sk'), plot(info_dp.dots(i,1), info_dp.dots(i,2), 'ok')
% legend('lower bound', 'upper bound', 'discrepit curve', 'discrepbil', 'discrepit')
% pause, hold off
% end
% 
% figure
% for i = [1:30, 31:5:80, 81:100]
% plot(info_sn_dpbil.DPcurvelow(i,:), '-b'), hold on, plot(info_sn_dpbil.DPcurveup(i,:), '-r'), plot(info_sn_dp.DPcurveup(i,:), '-g')
% plot(info_sn_dpbil.dots(i,1), info_sn_dpbil.dots(i,2), 'sk'), plot(info_sn_dp.dots(i,1), info_sn_dp.dots(i,2), 'ok')
% legend('lower bound', 'upper bound', 'discrepit curve', 'discrepbil', 'discrepit')
% pause, hold off
% end
