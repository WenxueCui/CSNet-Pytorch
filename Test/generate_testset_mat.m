clear;close all;
%% settings
folder = 'Set5';
scale = 1;
blocksize = 32;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));

for i = 1 : length(filepaths)        
    im_gt = imread(fullfile(folder,filepaths(i).name));
    if size(im_gt, 3) > 1
      im_gt = modcrop(im_gt, blocksize);
      im_gt = double(im_gt);
      im_gt_ycbcr = rgb2ycbcr(im_gt / 255.0);
      im_gt_y = im_gt_ycbcr(:,:,1) * 255.0;
      im_l_ycbcr = imresize(im_gt_ycbcr, 1/scale, 'bicubic');
      im_b_ycbcr = imresize(im_l_ycbcr, scale, 'bicubic');
      im_l_y = im_l_ycbcr(:,:,1) * 255.0;
      im_l = ycbcr2rgb(im_l_ycbcr) * 255.0;
      im_b_y = im_b_ycbcr(:,:,1) * 255.0;
      im_b = ycbcr2rgb(im_b_ycbcr) * 255.0;

      mat_output = [folder, '_mat2'];
      if exist(mat_output, 'dir') == 0
          mkdir(mat_output);
      end
      filename = [mat_output,'/',filepaths(i).name(1:end-4),'.mat'];
      save(filename, 'im_gt_y', 'im_b_y', 'im_l_y');
    else
      
      im_gt_y = im_gt;
      im_b_y = im_gt;
      im_l_y = im_gt;
      mat_output = [folder, '_mat2'];
      if exist(mat_output, 'dir') == 0
          mkdir(mat_output);
      end
      filename = [mat_output,'/',filepaths(i).name(1:end-4),'.mat'];
      save(filename, 'im_gt_y', 'im_b_y', 'im_l_y');
    end
end
