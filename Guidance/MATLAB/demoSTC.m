% Demo for paper "Fast Tracking via Spatio-Temporal Context Learning,"Kaihua Zhang, Lei Zhang,Ming-Hsuan Yang and David Zhang
% Paper can be available from http://arxiv.org/pdf/1311.1939v1.pdf 
% Implemented by Kaihua Zhang, Dept.of Computing, HK PolyU.
% Email: zhkhua@gmail.com
% Date: 11/24/2013
%%
clc;close all;clear;
%%
fftw('planner','patient')
%% set path
addpath('./demo_1');
img_dir = dir('./demo_1/*.jpg');
%% initialization
% data_
% initstate = [161,65,75,95];%initial rectangle [x,y,width, height]

% data
% initstate = [150,100,180,200];
% initstate = [150,120,100,100];

% data__
% initstate = [100,150,150,150];
% initstate = [50 130 200 200];
% initstate = [300,200,200,150];

% _data__
% initstate = [50, 360, 150, 150];

% __data__
% initstate = [360, 370, 120, 80];

% circ_circ
% initstate = [300 280 80 50];

% circ_zigzag
% initstate = [450 30 80 50];

% lingjian_zigzag
% initstate = [530 40 100 100];

% demo
% initstate = [140 40 20 30];

% demo_1
initstate = [150 40 20 25];

pos = [initstate(2)+initstate(4)/2 initstate(1)+initstate(3)/2];%center of the target
target_sz = [initstate(4) initstate(3)];%initial size of the target
%% parameters according to the paper
first = 1;
padding = 1;					%extra area surrounding the target
rho = 0.05;			        %the learning parameter \rho in Eq.(12)
sz = floor(target_sz * (1 + padding));% size of context region
%% parameters of scale update. See Eq.(15)
scale = 1;%initial scale ratio
lambda = 0.25;% \lambda in Eq.(15)
num = 5; % number of average frames
%% store pre-computed confidence map
alapha = 5;                    %parmeter \alpha in Eq.(6)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
dist = rs.^2 + cs.^2;
conf = exp(-0.5 * (sqrt(dist) / (alapha)) .^ 1);%confidence map function Eq.(6)
conf = conf/sum(sum(conf));% normalization
conff = fft2(conf); %transform conf to frequencey domain
%% store pre-computed weight window
hamming_window = hamming(sz(1)) * hann(sz(2))';
sigma = mean(target_sz);% initial \sigma_1 for the weight function w_{\sigma} in Eq.(11)
window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));% use Hamming window to reduce frequency effect of image boundary
window = window/sum(sum(window));%normalization
%%
for frame = first:numel(img_dir),
    sigma = sigma*scale;% update scale in Eq.(15)
    window = hamming_window.*exp(-0.5 / (sigma^2) *(dist));%update weight function w_{sigma} in Eq.(11)
    window = window/sum(sum(window));%normalization
 	%load image
    im = imread(img_dir(frame).name);	    
	if size(im,3) > 1,
		im = rgb2gray(im);
    end
   	contextprior = get_context(im, pos, sz, window);% the context prior model Eq.(4)
    %%
    if frame > first,
		%calculate response of the confidence map at all locations
	    confmap = real(ifft2(Hstcf.*fft2(contextprior))); %Eq.(11) 
       	%target location is at the maximum response
		[row, col] = find(confmap == max(confmap(:)), 1);
        pos = pos - sz/2 + [row, col]; 
        contextprior = get_context(im, pos, sz, window);
        conftmp = real(ifft2(Hstcf.*fft2(contextprior))); 
        maxconf(frame-1)=max(conftmp(:));
        %% update scale by Eq.(15)
        if (mod(frame,num+2)==0)
            scale_curr = 0;
            for kk=1:num
               scale_curr = scale_curr + sqrt(maxconf(frame-kk)/maxconf(frame-kk-1))
            end
%             scale = (1-lambda)*scale+lambda*(scale_curr/num)%update scale
                scale = scale
        end
        %%
    end	
	%% update the spatial context model h^{sc} in Eq.(9)
   	contextprior = get_context(im, pos, sz, window); 
    hscf = conff./(fft2(contextprior)+eps);% Note the hscf is the FFT of hsc in Eq.(9)
    %% update the spatio-temporal context model by Eq.(12)
    if frame == first,  %first frame, initialize the spatio-temporal context model as the spatial context model
		Hstcf = hscf;
	else
		%update the spatio-temporal context model H^{stc} by Eq.(12)
		Hstcf = (1 - rho) * Hstcf + rho * hscf;% Hstcf is the FFT of Hstc in Eq.(12)
    end
    %% visualization
    target_sz([2,1]) = target_sz([2,1])*scale;% update object size
	rect_position = [pos([2,1]) - (target_sz([2,1])/2), (target_sz([2,1]))];  
    imagesc(uint8(im))
    colormap(gray)
    rectangle('Position',rect_position,'LineWidth',4,'EdgeColor','r');
    hold on;
    text(5, 18, strcat('#',num2str(frame)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
    set(gca,'position',[0 0 1 1]); 
    pause(0.001); 
    hold off;
    drawnow;    
end