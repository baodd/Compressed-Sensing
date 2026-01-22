clear; close all; clc; rng(42);

%% ----- System parameters -----
Phi_half = 30*pi/180;
m  = -log(2)/log(cos(Phi_half));
Adet = 1e-4; Ts = 1; index = 1.5;
FOV  = 80*pi/180; G_Con = (index^2)/sin(FOV)^2;

lx=50; ly=50; lz=3;
nLED=25; N=nLED^2;
r=4;  M=200;  Nu=100;  Kmax=20; dth=5;

% LED grid
      [x_led,y_led]=meshgrid(linspace(0,lx,nLED));
      T_X=x_led(:)'; T_Y=y_led(:)'; T_Z=lz*ones(1,N);
      LED_pos=[T_X',T_Y',T_Z'];
%      [T_X,T_Y,T_Z,LED_pos] = make_irregular_leds(lx,ly,lz,nLED, ...
%                                 'jitter','jitter_frac',0.35,'seed',42);

%% ----- Groups (tile ~ 2r) -----
tile=2*r;
gx=max(1,floor(lx/tile)); gy=max(1,floor(ly/tile));
x_edges=linspace(0,lx,gx+1); y_edges=linspace(0,ly,gy+1);
groups={};
for ix=1:gx
  for iy=1:gy
    inX=(T_X>=x_edges(ix))&(T_X<x_edges(ix+1));
    inY=(T_Y>=y_edges(iy))&(T_Y<y_edges(iy+1));
    idx=find(inX&inY); if ~isempty(idx), groups{end+1}=idx(:); end
  end
end
if isempty(groups), groups={(1:N).'}; end

%% ----- Load optimized Sn, create random Sn -----
     [S_bin, Sn, infoS] = design_vlc_WCM_pure(M, N, groups, ...
          'pBern',0.1, 'steps',150000, 'alpha_intra',2.0, 'beta_inter',1.0, ...
          'seed',42, 'verbose',true, 'printEvery',2000);
    % S_bin: ma tran {0,1} 
     kNN = 8;                 % so láng gieng không gian xét trung bình
     polish_iters = 5000;     % so vòng hoán doi cap va tinh chinh (0 ?? t?t)
     [S, perm, infoR] = reorder_columns_by_geo_correlation(S_bin, LED_pos, kNN, polish_iters);
%       save('S_best.mat','S');    
%    load('S_best.mat','S');   % Sn ?ã t?i ?u (không chu?n hoá c?t)
Sn_opt = S;
Sn_rand = randi([0 1], M, N);
info_rand = compute_sensing_metrics(Sn_rand, groups, 2.0, 1.0);

fprintf('Sn_{rand}: mu_raw = %.4f | mu_weighted = %.4f | degCol ~= %.1f ± %.1f\n', ...
    info_rand.mu_raw, info_rand.mu_weighted, ...
    mean(info_rand.degCol), std(double(info_rand.degCol)));

LSn_opt  = power_method_norm(Sn_opt)^2;
LSn_rand = power_method_norm(Sn_rand)^2;

%% ----- Precompute for fairness -----
% 1) Tap vi trí UD co dinh cho moi SNR & moi ma tran
rng(5);
U_all = [rand(Nu,1)*lx, rand(Nu,1)*ly, zeros(Nu,1)];

% 2) Nhi?u chu?n hoá (chu?n N(0,1)), s? nhân v?i sigma t?ng tr??ng h?p
Z_all = randn(M, Nu);   % m?i c?t là 1 m?u nhi?u chu?n cho 1 UD

% 3) Kho?ng cách bình ph??ng ?? expand_neighborhood nhanh
XY = [T_X(:), T_Y(:)];
XX = sum(XY.^2,2);
D2 = bsxfun(@plus,XX,XX') - 2*(XY*XY.');
D2(D2<0)=0;

%% ----- Sweep SNR and evaluate both matrices -----
SNR_dB = 5:5:50; num_SNR = numel(SNR_dB);
MPE_opt = zeros(num_SNR,1); SRE_opt = zeros(num_SNR,1);
MPE_rand= zeros(num_SNR,1); SRE_rand= zeros(num_SNR,1);

t_all=tic;
for is=1:num_SNR
  snr_db = SNR_dB(is); snr = 10^(snr_db/10);
  [mpe1, sre1, hat_supp1, hat_u1] = eval_one_Sn(Sn_opt, LSn_opt, U_all, Z_all, snr_db);
  [mpe2, sre2, hat_supp2, hat_u2] = eval_one_Sn(Sn_rand,LSn_rand,U_all, Z_all, snr_db);
  MPE_opt(is)=mpe1; SRE_opt(is)=sre1;
  MPE_rand(is)=mpe2; SRE_rand(is)=sre2;
  fprintf('SNR=%2d dB | OPT: MPE=%.3f SRE=%.2f | RAND: MPE=%.3f SRE=%.2f\n',...
          snr_db, mpe1, sre1, mpe2, sre2);
end
fprintf('Total time = %.2f s\n', toc(t_all));

%% ----- Plot overlay -----
figure('Position',[100,100,900,340],'Color','w');

% --- MPE ---
subplot(1,2,1);
B1 = bar(SNR_dB, [MPE_opt, MPE_rand], 'grouped'); hold on;
B1(1).FaceColor = [0.30 0.60 0.95];   % GL
B1(2).FaceColor = [1.00 0.55 0.30];   % OMP
xlabel('SNR [dB]'); ylabel('MPE [m]');
title('Mean Position Error');
grid on; box on;
xticks(SNR_dB); xlim([min(SNR_dB)-1, max(SNR_dB)+1]);
legend({'Optimized matrix','Random matrix'},'Location','northeast');

% --- SRE ---
subplot(1,2,2);
B2 = bar(SNR_dB, [SRE_opt, SRE_rand], 'grouped'); hold on;
B2(1).FaceColor = [0.30 0.60 0.95];   % GL
B2(2).FaceColor = [1.00 0.55 0.30];   % OMP
xlabel('SNR [dB]'); ylabel('SRE');
title('Support Recovery Error');
grid on; box on;
xticks(SNR_dB); xlim([min(SNR_dB)-1, max(SNR_dB)+1]);
legend({'Optimized matrix','Random matrix'},'Location','northeast');

%% ----- 3D scene -----
if exist('hat_supp1','var') && ~isempty(hat_supp1)
    figure('Color','w','Name','VLC 3D - Estimated LEDs & UD'); 
    hold on; axis equal vis3d

    xlabel('X (m)','FontSize',9);
    ylabel('Y (m)','FontSize',9);
    zlabel('Z (m)','FontSize',9);
    grid on; view(145,25);

    xlim([0 lx]); ylim([0 ly]); zlim([0 lz*1.05]);
    drawRoomBox([0 lx],[0 ly],[0 lz],0.06);

    % ---- Plot LED and UD groups ----
    h_all = plot3(T_X(:),T_Y(:),T_Z(:),'o','MarkerSize',3, ...
                  'MarkerFaceColor',[0.7 0.7 0.7],'MarkerEdgeColor','none');     % All LEDs (gray)

    h_est_led = plot3(T_X(hat_supp1),T_Y(hat_supp1),T_Z(hat_supp1),'o','MarkerSize',3, ...
                      'MarkerFaceColor',[1 0.3 0.2],'MarkerEdgeColor','k');       % Estimated LEDs (red)

    h_est_ud = plot3(hat_u1(1),hat_u1(2),0,'o','MarkerSize',3, ...
                     'MarkerFaceColor',[0 0 1],'MarkerEdgeColor','k');            % Estimated UD (blue)

    camlight headlight; 
    lighting gouraud;

    ax = gca;
    ax.FontSize = 6;

    legend([h_all h_est_led h_est_ud], ...
           {'All LEDs','Estimated LEDs','Estimated User Device'}, ...
           'Location','northeast','FontSize',9);
end

%% ===================== Evaluation (GL pipeline rút g?n) =====================
function [mpe, sre, hat_supp, hat_u] = eval_one_Sn(Sn, LSn, U_all, Z_all, snr_db)
  % dùng ?úng các tham s? ngoài (via nested scope) — gi? s? t?n t?i:
  % lx,ly,lz,nLED,N,r,Kmax,dth,Nu,LED_pos,groups,m,Adet,Ts,G_Con,FOV, D2, M
  lx=evalin('base','lx'); ly=evalin('base','ly'); lz=evalin('base','lz');
  nLED=evalin('base','nLED'); N=evalin('base','N'); r=evalin('base','r');
  Kmax=evalin('base','Kmax'); dth=evalin('base','dth'); Nu=evalin('base','Nu');
  LED_pos=evalin('base','LED_pos'); groups=evalin('base','groups');
  m=evalin('base','m'); Adet=evalin('base','Adet'); Ts=evalin('base','Ts');
  G_Con=evalin('base','G_Con'); FOV=evalin('base','FOV'); M=evalin('base','M');
  D2=evalin('base','D2');

  snr = 10^(snr_db/10);
  errors=zeros(Nu,1); srev=zeros(Nu,1);

  rhoADMM = 1.0;
  AtA = Sn.'*Sn;
  cholL = chol(AtA + rhoADMM*speye(N) + 1e-12*speye(N),'lower');
  
  for iu=1:Nu
    u = U_all(iu,:);                % v? trí UD gi? c? ??nh gi?a hai ma tr?n

    % Kênh
    lambda=zeros(N,1); alpha=zeros(N,1);
    for i=1:N
      D=sqrt((u(1)-LED_pos(i,1))^2+(u(2)-LED_pos(i,2))^2+(u(3)-LED_pos(i,3))^2);
      cos_phi=(lz-u(3))/D; phi=acos(cos_phi);
      if hypot(u(1)-LED_pos(i,1),u(2)-LED_pos(i,2))<=r && phi<=FOV
        alpha(i)=(m+1)*Adet*cos_phi^(m+1)/(2*pi*D^2)*Ts*G_Con; lambda(i)=1;
      end
    end
    x_true=lambda.*alpha; true_support=find(x_true>0);

    % Tín hi?u & nhi?u công b?ng
    y0=Sn*x_true; p_sig=norm(y0)^2/M; sigma=sqrt(p_sig/snr);
    y = y0 + sigma*Z_all(:,iu);     % dùng cùng Z nh?ng scale theo sigma c?a Sn này

    % GL solve
    x_hat = group_lasso(Sn, y, groups, sigma, ...
             'rho',rhoADMM, 'CholL',cholL, 'AtA',AtA);
         
    % Debias NNLS (2 vòng)
    [~,o2]=sort(x_hat,'descend'); base=o2(1:min(6*Kmax,N));
    Uidx = base(:);
    for rad=[1.9,2.3]*r
        Uidx = unique([Uidx; expand_neighborhood_fast(Uidx,D2,(rad)^2,N)]);
    end
    x_hat=zeros(N,1); x_hat(Uidx)=lsqnonneg(Sn(:,Uidx),y);

    % V? trí UD
    [~, sidx] = sort(abs(x_hat),'descend');
    hat_supp = sidx(1); hat_u = LED_pos(hat_supp(1),1:2);
    for j=2:min(Kmax,length(sidx))
        z = LED_pos(sidx(j),1:2);
        if norm(z-hat_u) < dth
            hat_supp = [hat_supp; sidx(j)];
            hat_u = mean(LED_pos(hat_supp,1:2),1);
        end
    end

    errors(iu)=norm(hat_u - U_all(iu,1:2));
    srev(iu)=numel(setdiff(true_support,hat_supp))+numel(setdiff(hat_supp,true_support));
  end

  mpe=mean(errors); sre=mean(srev);
end

%% ===================== Helpers =====================
function Iout = expand_neighborhood_fast(seed,D2,radius2,N)
mask=false(N,1);
for k=1:numel(seed)
    i=seed(k);
    mask = mask | (D2(:,i) <= radius2);
end
Iout=find(mask);
end

function x_hat = group_lasso(S,y,groups,sigma,varargin)
p = inputParser;
addParameter(p,'NumLambda',20);
addParameter(p,'LambdaRatio',1e-4);
addParameter(p,'Nonneg',true);
addParameter(p,'Init',[]);
addParameter(p,'Tau',1.05);
addParameter(p,'FineRatio',1e-6);
addParameter(p,'FineNum',16);
% ADMM/LS
addParameter(p,'rho',1.0);
addParameter(p,'MaxIter',250);
addParameter(p,'Tol',1e-4);
addParameter(p,'Solver','chol');
addParameter(p,'LsqrTol',1e-6);
addParameter(p,'LsqrMaxit',500);
% FAST: nh?n AtA & Cholesky ?ã precompute
addParameter(p,'AtA',[]);
addParameter(p,'CholL',[]);
parse(p,varargin{:});

nlam=p.Results.NumLambda; ratio=p.Results.LambdaRatio;
nonneg=p.Results.Nonneg; x0=p.Results.Init;
tau=p.Results.Tau; fr=p.Results.FineRatio; fn=p.Results.FineNum;

[M,~]=size(S);
target=tau*sqrt(M)*sigma;

Sy=S.'*y;
lam_max=0;
for gi=1:numel(groups)
  g=groups{gi}; if isempty(g), continue; end
  lam_max=max(lam_max, norm(Sy(g),2));
end
lam_min=max(lam_max*ratio,1e-12);
lams=exp(linspace(log(lam_max),log(lam_min),nlam));

if isempty(x0), x=zeros(size(S,2),1); else, x=x0; end
x_best=x; lam_last=lam_max;

for il=1:nlam
  lam=lams(il); lam_last=lam;
  x = group_lasso_admm_core(S,y,groups,lam, ...
        'rho',p.Results.rho,'MaxIter',p.Results.MaxIter,'Tol',p.Results.Tol, ...
        'Nonneg',nonneg,'Init',x,'Solver',p.Results.Solver, ...
        'LsqrTol',p.Results.LsqrTol,'LsqrMaxit',p.Results.LsqrMaxit, ...
        'AtA',p.Results.AtA,'CholL',p.Results.CholL);
  x_best=x;
  if norm(y - S*x) <= target, break; end
end

lam2_min=max(lam_last*fr,1e-12);
lams2=exp(linspace(log(lam_last),log(lam2_min),fn));
x=x_best;
for il=1:length(lams2)
  lam=lams2(il);
  x = group_lasso_admm_core(S,y,groups,lam, ...
        'rho',p.Results.rho,'MaxIter',p.Results.MaxIter,'Tol',p.Results.Tol, ...
        'Nonneg',nonneg,'Init',x,'Solver',p.Results.Solver, ...
        'LsqrTol',p.Results.LsqrTol,'LsqrMaxit',p.Results.LsqrMaxit, ...
        'AtA',p.Results.AtA,'CholL',p.Results.CholL);
  x_best=x;
  if norm(y - S*x) <= target, break; end
end
x_hat=x_best;
end

function x = group_lasso_admm_core(S,y,groups,lambda,varargin)
p=inputParser;
addParameter(p,'rho',1.0);
addParameter(p,'MaxIter',250);
addParameter(p,'Tol',1e-4);
addParameter(p,'Nonneg',true);
addParameter(p,'Init',[]);
addParameter(p,'Solver','chol');
addParameter(p,'LsqrTol',1e-6);
addParameter(p,'LsqrMaxit',500);
% FAST: nh?n AtA & Cholesky ?ã precompute
addParameter(p,'AtA',[]);
addParameter(p,'CholL',[]);
parse(p,varargin{:});

rho = p.Results.rho; MaxIter=p.Results.MaxIter; Tol=p.Results.Tol;
nonneg=p.Results.Nonneg; x0=p.Results.Init;
solver=lower(p.Results.Solver);

[~,N]=size(S);
if ~isempty(p.Results.AtA)
    AtA = p.Results.AtA;
else
    AtA = S.'*S;
end
Aty = S.'*y;

if strcmp(solver,'chol')
    if ~isempty(p.Results.CholL)
        R = p.Results.CholL;      %%% FAST: tái dùng factor
    else
        R = chol(AtA + rho*speye(N) + 1e-12*speye(N),'lower');
    end
    solveM = @(b) R'\(R\b);
else
    M = AtA + rho*speye(N);
    solveM = @(b) lsqr(M,b,p.Results.LsqrTol,p.Results.LsqrMaxit);
end

if isempty(x0), x=zeros(N,1); else, x=x0; end
z=x; u=zeros(N,1);

gsize = cellfun(@numel, groups(:));
lamg  = lambda * sqrt(gsize(:));

for it=1:MaxIter
    x = solveM(Aty + rho*(z - u));
    v = x + u;
    z(:)=0;
    for gi=1:numel(groups)
        g=groups{gi}; if isempty(g), continue; end
        vg=v(g); ng=norm(vg,2);
        if ng>0
            shrink = max(0,1 - lamg(gi)/(rho*ng));
            z(g) = shrink*vg;
        end
    end
    if nonneg, z = max(z,0); end
    x_minus_z = x - z;
    u = u + x_minus_z;

    r_norm = norm(x_minus_z);
    s_norm = rho*norm(z - v);
    if r_norm < Tol*max(1,norm(x)) && s_norm < Tol*max(1,norm(u)), break; end
end
end

function s=power_method_norm(A,its)
if nargin<2, its=50; end
[~,n]=size(A); v=randn(n,1); nv=norm(v); if nv==0, v(1)=1; nv=1; end
v=v/nv;
for i=1:its, v=A.'*(A*v); nv=norm(v); if nv==0, break; end; v=v/nv; end
s=norm(A*v);
end

function [T_X,T_Y,T_Z,LED_pos] = make_irregular_leds(lx,ly,lz,nLED,mode,varargin)
% T?o l??i LED không ??u kho?ng cách nh?ng v?n có ?úng nLED x nLED ?i?m.
% mode = 'jitter'   : thêm nhi?u v? trí trong t?ng ô (an toàn v?i toàn b? code).
%      = 'nonlinear': nén/giãn to? ?? theo hàm m? ?? m?t ?? thay ??i theo không gian.
 
p = inputParser;
addParameter(p,'jitter_frac',0.35);   % t? l? t?i ?a so v?i kích th??c ô (0..0.49)
addParameter(p,'alphaX',1.5);         % h? s? cong cho 'nonlinear' theo tr?c X
addParameter(p,'alphaY',1.5);         % h? s? cong cho 'nonlinear' theo tr?c Y
addParameter(p,'seed',42);
parse(p,varargin{:});
rng(p.Results.seed);
 
% L??i ??u c? s? (ch? ?? ?ánh s? index/reshape)
[xg,yg] = meshgrid(linspace(0,lx,nLED), linspace(0,ly,nLED));
 
switch lower(mode)
  case 'jitter'
    % kích th??c ô
    dx = lx/(nLED-1+eps);
    dy = ly/(nLED-1+eps);
    a  = min(0.49, max(0, p.Results.jitter_frac));  % tránh v??t biên/??ng nhau quá nhi?u
    % jitter ??c l?p cho t?ng ?i?m
    Jx = (2*rand(size(xg))-1) * (a*dx);
    Jy = (2*rand(size(yg))-1) * (a*dy);
    % không jitter ? biên quá m?c: gi?m d?n v? 0 sát mép
    wx = min(xg/dx, (lx-xg)/dx); wx = min(1, max(0, wx));
    wy = min(yg/dy, (ly-yg)/dy); wy = min(1, max(0, wy));
    X = xg + wx.*Jx;  X = min(max(X,0), lx);
    Y = yg + wy.*Jy;  Y = min(max(Y,0), ly);
 
  case 'nonlinear'
    % ánh x? phi tuy?n (0..1) -> (0..1): u -> u^alpha / (u^alpha+(1-u)^alpha)
    % alpha>1: dày ? gi?a; 0<alpha<1: dày ? rìa
    aX = p.Results.alphaX; aY = p.Results.alphaY;
    ux = linspace(0,1,nLED);
    uy = linspace(0,1,nLED);
    fx = @(u,a) (u.^a) ./ (u.^a + (1-u).^a + eps);
    fy = fx;
    X = lx * fx(ux,aX);  Y = ly * fy(uy,aY);
    [X,Y] = meshgrid(X,Y);
    % thêm jitter nh? ?? phá ??ng hàng (gi? biên)
    dx = lx/(nLED-1+eps); dy = ly/(nLED-1+eps);
    Jx = 0.08*dx*(2*rand(size(X))-1);
    Jy = 0.08*dy*(2*rand(size(Y))-1);
    X = min(max(X+Jx,0),lx);  Y = min(max(Y+Jy,0),ly);
 
  otherwise
    error('Unknown mode');
end
 
T_X = X(:)'; T_Y = Y(:)'; T_Z = lz*ones(1, numel(T_X));
LED_pos = [T_X', T_Y', T_Z'];
end

function drawRoomBox(xrng,yrng,zrng,alphaFace)
[x0,x1]=deal(xrng(1),xrng(2)); [y0,y1]=deal(yrng(1),yrng(2)); [z0,z1]=deal(zrng(1),zrng(2));
faces={ [x0 x1 x1 x0; y0 y0 y1 y1; z0 z0 z0 z0], [x0 x1 x1 x0; y0 y0 y1 y1; z1 z1 z1 z1], ...
        [x0 x1 x1 x0; y0 y0 y0 y0; z0 z0 z1 z1], [x0 x1 x1 x0; y1 y1 y1 y1; z0 z0 z1 z1], ...
        [x0 x0 x0 x0; y0 y1 y1 y0; z0 z0 z1 z1], [x1 x1 x1 x1; y0 y1 y1 y0; z0 z0 z1 z1]};
col=[0.85 0.95 1];
for f=1:numel(faces), p=faces{f};
  patch(p(1,:),p(2,:),p(3,:),col,'FaceAlpha',alphaFace,'EdgeColor',[0.7 0.8 0.9]);
end
end

function [S_bin, Sn, info] = design_vlc_WCM_pure(M, N, groups, varargin)
p = inputParser;
addParameter(p,'pBern',0.35);
addParameter(p,'steps',3e4);
addParameter(p,'alpha_intra',2.0);
addParameter(p,'beta_inter',1.0);
addParameter(p,'seed',[]);
addParameter(p,'verbose',true);
addParameter(p,'printEvery',2000);
parse(p,varargin{:});

pBern      = p.Results.pBern;
steps      = p.Results.steps;
alpha_intra= p.Results.alpha_intra;
beta_inter = p.Results.beta_inter;
seed       = p.Results.seed;
verbose    = p.Results.verbose;
printEvery = p.Results.printEvery;

if ~isempty(seed), rng(seed); end

% ---- 1) Khoi tao S voi bac cot gan nhu co dinh (giu dúng trong refine)
deg_col_target = max(1, round(pBern * M));    % bac muc tiêu moi cot
S = false(M,N);
for j = 1:N
    idx = randperm(M, deg_col_target);
    S(idx,j) = true;
end
S = logical(S);

% ---- 2) Chuan bi trong so W theo nhóm ----
group_id = zeros(1,N);
if ~isempty(groups)
    for gi = 1:numel(groups)
        group_id(groups{gi}) = gi;
    end
    % Phòng cot nào không duoc gán nhóm
    group_id(group_id==0) = max(group_id)+1;
else
    group_id(:) = 1;
end

W = ones(N,N)*beta_inter;
for i = 1:N
    for j = i+1:N
        if group_id(i)==group_id(j), w = alpha_intra; else, w = beta_inter; end
        W(i,j)=w; W(j,i)=w;
    end
end
W(1:N+1:end) = 0;   % zero trên duong chéo (không xét i=j)

% ---- 3) Gram ban dau (không chuan hoá), O = off-diagonal ----
G = double(S).' * double(S);   % N×N
for i=1:N, G(i,i)=0; end
AbsW = W .* abs(G);

% Duy trì "max theo hàng" de cap nhat nhanh
rowMaxVal = zeros(N,1);
rowMaxIdx = ones(N,1);
for i=1:N
    [rowMaxVal(i), rowMaxIdx(i)] = max(AbsW(i,:));
end
best_val = max(rowMaxVal);     % = max_{i?j} w_ij * |<s_i, s_j>|
bestS = S; bestG = G; bestRowMaxVal=rowMaxVal; bestRowMaxIdx=rowMaxIdx;
bestStep = 0;

t0 = tic;
for it = 1:steps
    % ---- 4) Chon cot ngau nhiên và hoán vi 0<->1 de giu bac cot ----
    c = randi(N);
    z = find(~S(:,c)); o = find(S(:,c));
    if isempty(z) || isempty(o), continue; end
    r0 = z(randi(numel(z)));   % vi trí 0 -> 1
    r1 = o(randi(numel(o)));   % vi trí 1 -> 0

    % vector hàng tai hai dòng liên quan
    row0 = S(r0,:);    % 1×N logic
    row1 = S(r1,:);    % 1×N logic

    % ---- 5) Cap nhat Gram hàng/cot c theo gia tang (chi thay doi <s_c, s_j>) ----
    % Voi S(:,c) thay doi 2 bit, <s_c, s_j> m?i = c? + row0(j) - row1(j)
    Gc_new = G(c,:);                 % 1×N
    delta  = double(row0) - double(row1);
    Gc_new = Gc_new + delta;         % cap nhat hàng
    Gc_new(c) = 0;

    % L?u t?m ?? tính max nhanh
    AbsW_c_new = AbsW(c,:);
    AbsW_c_new = abs(Gc_new) .* W(c,:);

    % ---- 6) Danh gia coherence co trong so moi mot cách tuyen tính ----
    % Can cap nhat max theo hàng cho:
    %  - Hàng c (moi)
    %  - Các hàng k?c mà ph?n t? (k,c) b? ??i
    rowMaxVal_trial = rowMaxVal; 
    rowMaxIdx_trial = rowMaxIdx;

    % Hàng c:
    [rowMaxVal_trial(c), rowMaxIdx_trial(c)] = max(AbsW_c_new);

    % Các hàng k ? c:
    for k = 1:N
        if k==c, continue; end
        % Giá tri (k,c) moi:
        new_kc = abs(Gc_new(k)) * W(k,c);
        old_kc = AbsW(k,c);
        if new_kc >= rowMaxVal_trial(k)
            rowMaxVal_trial(k) = new_kc; rowMaxIdx_trial(k) = c;
        else
            if rowMaxIdx_trial(k)==c && new_kc < old_kc
                % Max c? c?a hàng k n?m ? c ? ph?i quét l?i c? hàng k
                rowLine = AbsW(k,:);
                rowLine(c) = new_kc;             % thay b?ng giá tr? m?i
                [rowMaxVal_trial(k), rowMaxIdx_trial(k)] = max(rowLine);
            end
        end
    end

    trial_best = max(rowMaxVal_trial);

    % ---- 7) Chap nhan neu không te hon (giam max là lý tuong) ----
    if trial_best <= best_val
        % commit thay ??i
        S(r0,c) = true; S(r1,c) = false;

        % c?p nh?t G và AbsW cho hàng/c?t c
        for k=1:N
            if k==c, continue; end
            G(k,c) = Gc_new(k);
            G(c,k) = Gc_new(k);
            AbsW(k,c) = abs(G(k,c)) * W(k,c);
            AbsW(c,k) = AbsW(k,c);
        end
        % hàng c (???ng chéo = 0)
        G(c,c) = 0; AbsW(c,c)=0;

        % c?p nh?t rowMaxVal/Idx chính th?c
        rowMaxVal = rowMaxVal_trial;
        rowMaxIdx = rowMaxIdx_trial;

        % c?p nh?t best n?u t?t h?n
        if trial_best < best_val
            best_val = trial_best;
            bestS = S; bestG = G;
            bestRowMaxVal=rowMaxVal; bestRowMaxIdx=rowMaxIdx;
            bestStep = it;
        end
    end

    if verbose && printEvery>0 && mod(it,printEvery)==0
        fprintf('  WCM step %6d/%6d, w-coh=%.4f (best at %d)\n', ...
            it, steps, best_val, bestStep);
    end
end

S_bin = bestS;
Sn    = double(S_bin);  % KHÔNG CHU?N HOÁ theo yêu c?u

% Th?ng kê
Gbest = bestG;
mu_raw = max(max(abs(Gbest - diag(diag(Gbest)))));       % coherence thô (không tr?ng s?)
mu_w   = max(bestRowMaxVal);                             % coherence có tr?ng s?
degCol = full(sum(S_bin,1));
degRow = full(sum(S_bin,2));

info.mu_raw       = mu_raw;
info.mu_weighted  = mu_w;
info.steps_done   = steps;
info.time         = toc(t0);
info.deg_col      = degCol;
info.deg_row      = degRow;
info.alpha_intra  = alpha_intra;
info.beta_inter   = beta_inter;
info.pBern        = pBern;

if verbose
    fprintf('[WCM] done in %.2fs | mu_w=%.4f | mu_raw=%.4f | Col~=%d±%.1f\n', ...
        info.time, info.mu_weighted, info.mu_raw, round(mean(degCol)), std(double(degCol)));
end
end

function [S, perm, info] = reorder_columns_by_geo_correlation(S, LED_pos, kNN, polish_iters)
t0 = tic; S = double(S); [~,N] = size(S); XY = LED_pos(:,1:2);
kNN = max(1,min(kNN,N-1)); polish_iters = max(0,round(polish_iters));

% --- x?p h?ng theo t??ng quan & kho?ng cách ---
C  = abs(S.'*S); C(1:N+1:end) = 0;
C2 = C;         C2(1:N+1:end) = inf;
[~, order_cols] = sort(mean(mink(C2,kNN,2),2),'ascend');

XX = sum(XY.^2,2);
D  = sqrt(bsxfun(@plus,XX,XX') - 2*(XY*XY')); D(1:N+1:end) = inf;
[~, order_led]  = sort(mean(mink(D,kNN,2),2),'ascend');

% --- ghép 1-1 & sensing sau reorder ---
perm = zeros(1,N); perm(order_led) = order_cols; S_perm = S(:,perm);

% --- t?o c?nh kNN theo hình h?c (dùng cho loss) ---
[~,nn] = mink(D,kNN,2);
E = unique(sort([repmat((1:N)',kNN,1) nn(:)],2),'rows');
loss = @(p,ee) sum( C(sub2ind([N,N], p(E(ee,1)), p(E(ee,2))) ) );

% --- polish b?ng hoán v? c?c b? ---
best_perm = perm; best_loss = loss(best_perm,1:size(E,1));
if polish_iters > 0
  inc = cell(N,1);
  for e = 1:size(E,1), i=E(e,1); j=E(e,2); inc{i}=[inc{i};e]; inc{j}=[inc{j};e]; end
  for it = 1:polish_iters
    i = randi(N); j = randi(N); if i==j, continue; end
    ee = unique([inc{i}; inc{j}]);
    old = loss(best_perm,ee);
    p   = best_perm; p([i j]) = p([j i]);
    new = loss(p,ee);
    if new <= old, best_perm = p; best_loss = best_loss - old + new; end
  end
  perm   = best_perm;
  S = S(:,perm);
end

info = struct('time_sec',toc(t0),'loss_final',best_loss,'kNN',kNN,'polish',polish_iters);
% fprintf('[geo-reorder] kNN=%d | polish=%d | loss=%.2f | time=%.2fs\n', kNN, polish_iters, best_loss, info.time_sec);
end

function infoM = compute_sensing_metrics(Sn, groups, alpha_intra, beta_inter)
% COMPUTE_SENSING_METRICS
%   Tính m?t s? ch? s? c?a ma tr?n ?o nh? phân Sn:
%   - ?? t??ng quan c?c ??i (mu_raw)
%   - ?? t??ng quan có tr?ng s? (mu_weighted)
%   - S? bit '1' trên m?i c?t (degCol)

    [M, N] = size(Sn);

    % --- 1) Ma tr?n Gram (không chu?n hóa) ---
    G = double(Sn).' * double(Sn);   % N x N
    G(1:N+1:end) = 0;               % b? ???ng chéo

    % --- 2) ?? t??ng quan c?c ??i (raw coherence) ---
    mu_raw = max(max(abs(G)));      % max_{i?j} |<s_i, s_j>|

    % --- 3) Xây ma tr?n tr?ng s? W d?a trên groups ---
    group_id = zeros(1,N);
    if ~isempty(groups)
        for gi = 1:numel(groups)
            group_id(groups{gi}) = gi;
        end
        group_id(group_id==0) = max(group_id)+1;
    else
        group_id(:) = 1;
    end

    W = ones(N,N)*beta_inter;
    for i = 1:N
        for j = i+1:N
            if group_id(i)==group_id(j)
                w = alpha_intra;    % cùng nhóm ? tr?ng s? l?n
            else
                w = beta_inter;     % khác nhóm
            end
            W(i,j) = w;
            W(j,i) = w;
        end
    end
    W(1:N+1:end) = 0;

    % --- 4) ?? t??ng quan có tr?ng s? (weighted coherence) ---
    AbsW = W .* abs(G);          % w_ij * |G_ij|
    mu_w = max(AbsW(:));         % max_{i?j} w_ij |G_ij|

    % --- 5) S? bit '1' m?i c?t ---
    degCol = full(sum(Sn,1));    % 1 x N

    % --- 6) Gói vào struct k?t qu? ---
    infoM = struct();
    infoM.mu_raw      = mu_raw;
    infoM.mu_weighted = mu_w;
    infoM.degCol      = degCol;
    infoM.M           = M;
    infoM.N           = N;
end
