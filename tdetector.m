%% TDETECTOR Step Detection Algorithm (Tdetector1 and Tdetector2)
% [Y,out] = tdetector(X,var_option)
%
% REQUIRED INPUT:
% -------------------------------------------------------------------
% X: vector of a piecewise constant function hidden in white noise
%
% OPTIONAL INPUT:
% -------------------------------------------------------------------
% var_option:
%   [1] assume the underlying variance of X is constant throughout (default)
%   [2] assume the underlying variance of X changes throughout
%
% OUTPUTS:
% -------------------------------------------------------------------
% Y: column vector showing step function fit of X (same length as X)
% out: structure containing information of fitting
%   di: column vector of the declared step indexes (index of each new plateau)
%   ssz: column vector of each step size
%   psz: column vector of each plateau size
%   vx: column vector of the variance at each index
% 
% for additional info, visit:
% <a href="matlab:web('http://www.bioe.psu.edu/labs/Hancock-Lab/tdetector.html','-browser')">http://www.bioe.psu.edu/labs/Hancock-Lab/tdetector.html</a>

% NOTES:
% -------------------------------------------------------------------
% - The "out" structure output and the "var_option" input do not have to be
%   included when calling the function. Y = tdetector(X); is valid.
%
% - Cell titles for secondary functions in the code below include the 
%   var_options that utilize that respective function in parentheses.

% EXAMPLES:
% -------------------------------------------------------------------
% 1. Demonstrate Tdetector1:
%
% X = randn(1000,1);
% X(200:end) = X(200:end) + 5;
% X(400:end) = X(400:end) + 5;
% X(600:end) = X(600:end) + 5;
% plot(X,'b'); hold on
% [Y,out] = tdetector(X);
% disp('declared step indexes:');disp(out.di)
% plot(Y,'g');
%
% 2. Demonstrate Tdetector2:
%
% X = [1*randn(199,1);2*randn(200,1);3*randn(200,1);8*randn(401,1);];
% X(200:end) = X(200:end) + 5;
% X(400:end) = X(400:end) + 5;
% X(600:end) = X(600:end) + 5;
% plot(X,'b'); hold on
% [Y,out] = tdetector(X,2);
% disp('declared step indexes:');disp(out.di)
% plot(Y,'g');

% Nathan Deffenbaugh
% ncd50561234@gmail.com, ncd5056@psu.edu
% (2014 June 10)

%% tdetector

function [Y,out] = tdetector(X,var_option)

% check the var_option, store as VO
VO = 1; % (default to constant variance)
if exist('var_option','var')
    if var_option == 2
        VO = 2;
    end
end

% define full data length
Lo = length(X);

% calculate underlying variance or variance sections (vx)
if VO == 1
    SIG = getSig(X,1);
    out.vx = SIG^2*ones(Lo,1);
else
    vx = varSect(X);
    out.vx = vx';
end

% define the empirical multiplier lookup table and linearly interpolate
multTab = [1,0;2,2;3,2.17;4,2.34;6,2.47;8,2.60;11,2.656250;16,2.75;23,2.815625;32,2.90;45,2.940625;64,3;91,3.0421875000;128,3.10;181,3.1207031250;256,3.15;362,3.19754716981100;512,3.24;724,3.280080;1024,3.304768;1448,3.318336;2048,3.325240;2896,3.329480;4096,3.331096;5793,3.332793;8192,3.3331644000;1e4,3.333300;1e10,3.333300];
multTab = interp1(multTab(:,1),multTab(:,2),1:Lo);

% step detecting loop
plats_array = [1,Lo];
found = [];
while ~isempty(plats_array)
    Bound = plats_array(end,:);
    % look for a step in this current section of the data
    if VO == 1
        [step_index,status] = detectStep1(X(Bound(1):Bound(2)),Bound(1),SIG, multTab);
    else
        [step_index,status] = detectStep2(X(Bound(1):Bound(2)),Bound(1),vx(Bound(1):Bound(2)), multTab);
    end
    % if a significant step is detected 
    if status == 1
        found(end+1,1) = step_index;
        plats_array(end+1,:) = [plats_array(end,1),step_index-1];
        plats_array(end+1,:) = [step_index,plats_array(end-1,2)];
        plats_array(end-2,:) = [];
    elseif status == -1
        plats_array(end,:) = [];
    end
end

% sort the found steps
found = [found;1;Lo];
found = sortrows(found);

% check found steps and build Y vector
if VO == 1
    [checked] = checkSteps(found, X, SIG, multTab, VO);
else
    [checked] = checkSteps(found, X, vx, multTab, VO);
end
checked = [1;checked;Lo+1];
for ii = 1:(length(checked)-1)
    Y(checked(ii):checked(ii+1)-1) = mean(X(checked(ii):checked(ii+1)-1));
end
Y = Y';

% calculate step sizes
step_sizes = zeros(length(checked) - 2,1);
for ii = 2:length(checked)-1
    step_sizes(ii-1) = Y(checked(ii)) - Y(checked(ii) - 1);
end
out.ssz = step_sizes;

% calculate plateau sizes (how many indexes exist between each found step)
out.psz = checked(2:end) - checked(1:end-1);

% output declared step indexes
out.di = checked(2:end-1);

end

%% getSig (1,2)

function [SIG] = getSig(Xs,expnt)

% define pairwise difference vectors
diff1 = diff(Xs);
diff2 = diff(Xs).^2;

while true
    % current estimate of sigma of X
    sigmaC = (mean(diff2)/2)^0.5;
    
    % remove outlier values from diff vectors
    diff2(abs(diff1) > 3*(2^.5)*sigmaC) = [];
    diff1(abs(diff1) > 3*(2^.5)*sigmaC) = [];
    
    % new estimate of sigma of X
    sigmaN = (mean(diff2)/2)^0.5;
    
    if sigmaN == sigmaC
        break
    end
end
% empirical correction for underestimation
SIG = sigmaN*1.015;
SIG = SIG^expnt;

end

%% getSigLoop (2)
% getSig function altered slightly to improve speed during a loop

function [SIG] = getSigLoop(Xs,expnt,sd2,ld2)

% Sig Equation
diff1 = diff(Xs);
diff2 = diff(Xs).^2;

% initiate sum to subtract
sts = 0;
% initiate length to subtract
lts = 0;

STOP = 0;
while (STOP == 0)
    SIG = ((sd2-sts)/(ld2-lts)/2)^0.5;
  
    sd2 = sd2-sts;
    ld2 = ld2-lts;
    
    icurrpeaks = (abs(diff1) > (3*(2^.5))*SIG);
    currpeaks = diff2(icurrpeaks);
    % sum to subtract
    sts = sum(currpeaks);
    % length to subtract
    lts = length(currpeaks);
    
    % zero out peaks
    diff1(icurrpeaks) = 0;
    
    if (lts == 0)
        break
    end
end
SIG = SIG*1.015;
SIG = SIG^expnt;

end

%% varSect (2)

function [vx] = varSect(X)

% define full data length
Lo = length(X);

% variance sectioning loop
plats_array = [1,Lo];
found = [];
while ~isempty(plats_array)
    Bound = plats_array(end,:);
    [step_index,status] = detectVars(X(Bound(1):Bound(2)),Bound(1));
    
    if status == 1
        found(end+1,1) = step_index;
        plats_array(end+1,:) = [plats_array(end,1),step_index-1];
        plats_array(end+1,:) = [step_index,plats_array(end-1,2)];
        plats_array(end-2,:) = [];
    elseif status == -1
        plats_array(end,:) = [];
    end
end

% sort the found variance steps
found = [found;1;Lo];
found = sortrows(found);

% check found variance steps
[checked] = checkVars(found,X);
checked = [1;checked;Lo+1];

% build vx vector
for ii = 1:(length(checked)-1)
    vx(checked(ii):checked(ii+1)-1) = getSig(X(checked(ii):checked(ii+1)-1),2);
end

end

%% detectVars (2)

function [mxi,status] = detectVars(Xs,i_1)
% in order for any pairwise difference value to be > z*(2^.5)sig and hence excluded,
%  the length, n, of the diff vector must be n >= z^2 + 1; L >= 2(z^2 + 2). For 
%  z = 3, L >= 22. Requiring L >= 22 is necessary to ensure that large pairwise
%  differences due to true steps in the data do not influence the calculated
%  variance of that section.

% define L and default values
L = length(Xs);
status = -1;
mxi = 0;

if (L >= 22)
    d2 = diff(Xs).^2;

    % get sigma of noise
    SIG = getSig(Xs,1);

    % DOV significance rating
    Asd2 = sum(d2(1:9));
    Bsd2 = sum(d2(11:end));
    RVD = zeros(L-2,1);
    for ii = 11:L-10
        Asd2 = Asd2 + d2(ii-1);
        Bsd2 = Bsd2 - d2(ii);

        A = Xs(1:ii);
        B = Xs(ii+1:end);

        VA = getSigLoop(A,2,Asd2,ii-1);
        VB = getSigLoop(B,2,Bsd2,L-ii-1);
        DOV = VA - VB;
        
        LA = ii;
        LB = L - ii;
        sigma_squared = ( ((LA^2 + LA -3)/((LA-1)^2)) + ((LB^2 + LB -3)/((LB-1)^2)) - 2 )*SIG^4;
        
        RVD(ii) = DOV/(sigma_squared^.5)/3;

    end

    % find the index of the max RVD (rated difference of variance)
    mxi = find(abs(RVD) == max(abs(RVD)));
    mxi = max(mxi);

    % determine the status of the section based on the RVD
    if (abs(RVD(mxi)) > 1)
        status = 1;
        % globalize the step index
        mxi = mxi + i_1;   
    end
end

end

%% detectStep1 (1)

function [mxi,status] = detectStep1(Xs,i_1,SIG,multTab)

% define L and default values
L = length(Xs);
status = -1;
mxi = 0;

if (L >= 2)

    % declare sigma multiplier
    mult = multTab(L);
    % Chip's arbitrary increase in mult
    mult = mult * 3;

    % DOM significance rating
    RMD = zeros(L,1);
    m1 = 0;
    m2 = sum(Xs);
    for ii = 1:L-1

        m1 = m1 + Xs(ii);
        m2 = m2 - Xs(ii);
        DOM = m2/(L-ii) - m1/(ii);
        sigma = SIG*(1/ii + 1/(L-ii))^.5;

        RMD(ii+1) = DOM/(sigma*mult);

    end

    % find the index of the max RMD (rated difference of mean)
    mxi = find(abs(RMD) == max(abs(RMD)));
    mxi = max(mxi);

    % determine the status of the section based on the RMD
    status = -1;
    if (abs(RMD(mxi)) > 1)
        status = 1;
        % globalize the step index
        mxi = mxi + i_1 - 1;   
    end
end

end

%% detectStep2 (2)

function [mxi,status] = detectStep2(Xs,i_1,vx,multTab)

% define L and default values
L = length(Xs);
status = -1;
mxi = 0;

if (L >= 2)

    % get sigma of noise
    SIG = mean(vx)^.5;

    % declare sigma multiplier
    mult = multTab(L);
    % Chip's arbitrary increase in mult
    mult = mult * 3;

    % DOM significance rating
    RMD = zeros(L,1);
    m1 = 0;
    m2 = sum(Xs);
    for ii = 1:L-1

        m1 = m1 + Xs(ii);
        m2 = m2 - Xs(ii);
        DOM = m2/(L-ii) - m1/(ii);
        sigma = SIG*(1/ii + 1/(L-ii))^.5;

        RMD(ii+1) = DOM/(sigma*mult);

    end

    % find the index of the max RMD (rated difference of mean)
    mxi = find(abs(RMD) == max(abs(RMD)));
    mxi = max(mxi);

    % define the RMD_vx value
    sigma_vx = (( sum(vx(1:(mxi - 1)))/((mxi - 1)^2) ) + ( sum(vx((mxi):end))/((L-mxi+1)^2) ))^.5;
    DOM = mean(Xs(1:(mxi - 1))) - mean(Xs(mxi:end));
    RMD_vx = DOM/(mult*sigma_vx);

    % determine the status of the section based on the RMD and RMD_vx
    status = -1;
    if (abs(RMD(mxi)) > 1 && abs(RMD_vx) > 1)
        status = 1;
        % globalize the step index
        mxi = mxi + i_1 - 1;   
    end
end

end

%% checkVars (2)

function [checked] = checkVars(found, rx)

% shift last index
found(end) = found(end) + 1;

% initialize
checked = [];
cc = 0;
endW = 0;

% if there are no found sections to check, do not enter checking loop
if (length(found) == 2)
    endW = 1;
end

% variance section checking loop
ii = 1;
while (endW == 0)
    ii = ii + 1;
    % check variance steps now based on adjacent variance plateaus
    [step_index,status] = detectVars(rx(found(ii-1):found(ii+1)-1),found(ii-1));
    
    % if the step is still significant based on adjacent plateaus, store it to the checked vector
    if (status == 1)
        cc = cc + 1;
        checked(cc,1) = step_index;
    % else, remove it from the found vector
    else
        found(ii) = [];
        ii = ii - 1;        
    end
    
    % if there are no more steps to check, end this while loop
    if ((ii + 1) == length(found))
        endW = 1;
    end
end

end

%% checkSteps (1,2)

function [checked] = checkSteps(found, rx, noise_input, multTab, VO)

% store noise_input
if VO == 1
    SIG = noise_input;
else
    vx = noise_input;
end

% shift last index
found(end) = found(end) + 1;

% initialize
checked = [];
cc = 0;
endW = 0;

% if there are no found steps to check, do not enter checking loop
if (length(found) == 2)
    endW = 1;
end

% step checking loop
ii = 1;
while (endW == 0)
    ii = ii + 1;
    % check steps now based on adjacent plateaus (depending on variance option, VO)
    if VO == 1
        [step_index,status] = detectStep1(rx(found(ii-1):found(ii+1)-1),found(ii-1),SIG, multTab);
    else
        [step_index,status] = detectStep2(rx(found(ii-1):found(ii+1)-1),found(ii-1),vx(found(ii-1):found(ii+1)-1), multTab);
    end
    
    % if the step is still significant based on adjacent plateaus, store it to the checked vector
    if (status == 1)
        cc = cc + 1;
        checked(cc,1) = step_index;
    % else, remove it from the found vector
    else
        found(ii) = [];
        ii = ii - 1;
    end
    
    % if there are no more steps to check, end this while loop
    if ((ii + 1) == length(found))
        endW = 1;
    end
end

end
